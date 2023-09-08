
import * as d3 from "d3";

import './style.css';
import { canvas, get_scales, get_svg, stroke_color, highlight_stroke_color, boxDrawProps } from "./canvas";
import { BinstBox, Cxn, Soquet, RegInstance } from "./binst";

/* Set up drawing surface using function in the canvas module */
const { x, x_hat, y, y_hat } = get_scales(canvas);
const svg = get_svg(canvas, x, y)

let BOXES: Array<BinstBox> = [];
let CXNS: Array<Cxn> = [];

interface BinstBoxScreenCoords {
  x: number;
  y: number;
}

type TransitionIn = d3.Transition<d3.BaseType, any, any, any>;


function get_box_drag_behavior(): d3.DragBehavior<SVGGElement, BinstBox, BinstBoxScreenCoords> {

  function dragstarted(event: any, d: BinstBox) {
    d3.select(this).raise().selectAll("rect").attr("stroke", highlight_stroke_color);
    if(!event.active) simulation.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
  }

  function dragged(event: any, d: BinstBox) {
    d.fx = x.invert(event.x);
    d.fy = y.invert(event.y);
    /*d3.select(this).attr("transform", `translate(${event.x}, ${event.y})`)*/
    /*update_cxns(CXNS);*/
  }

  function dragended(event: any, d: BinstBox) {
    d3.select(this).selectAll("rect").attr("stroke", stroke_color);
    if(!event.active) simulation.alphaTarget(0);
    d.fx = null;
    d.fy = null;
  }

  function subject(event: any, d: BinstBox): BinstBoxScreenCoords {
    return { x: x(d.x), y: y(d.y) }
  }

  return d3.drag<SVGGElement, BinstBox, BinstBoxScreenCoords>()
    .subject(subject)
    .on("start", dragstarted)
    .on("drag", dragged)
    .on("end", dragended);
}

function soq_y(i: number): number {
  /* note: in user coordinates */
  return (boxDrawProps.headerHeight + boxDrawProps.perSoqHeight * (i + 0.5))
}

function cxn_key(cxn: Cxn): string {
  let [left, right] = cxn;
  return `${left.binst.i}_${left.soq_i}_${right.binst.i}_${right.soq_i}`
}

function apply_cxn_pos_attrs(sel: any) {
  return sel
    /* x1 and y1 are the "from" which is the right side of the box */
    .attr("x1", (d: Cxn) => x(d[0].binst.x + boxDrawProps.width))
    .attr("y1", (d: Cxn) => y(d[0].binst.y + soq_y(d[0].soq_i)))
    /* x2 and y2 are the "to" which is the left side of the box */
    .attr("x2", (d: Cxn) => x(d[1].binst.x + 0))
    .attr("y2", (d: Cxn) => y(d[1].binst.y + soq_y(d[1].soq_i)))
}

function d3_join_cxns(cxns: Array<Cxn>, tt: TransitionIn) {
  return svg.selectAll("line.cxn")
    .data(cxns, cxn_key)
    .join(
      enter => enter.append("line")
        .attr("class", "cxn")
        .attr("stroke", stroke_color)
        .attr("stroke-width", 2)
        .attr("opacity", 0.0)
        .call(apply_cxn_pos_attrs)
        .call(enter => enter.transition(tt)
          .attr("opacity", 1.0)
        ),
      update => update
        .call(update => {
          let sel = update.transition(tt);
          apply_cxn_pos_attrs(sel);
        }),
      exit => exit
        .call(exit => exit.transition(tt)
          .attr("opacity", 0.0)
          .remove()
        ),
    )
}


const THRU = "Side.THRU";
const LEFT = "Side.LEFT";
const RIGHT = "Side.RIGHT";


function add_soqlabels(g: d3.Selection<SVGGElement, BinstBox, SVGSVGElement, undefined>) {
  g.selectAll("text.soqlabel")
    .data((d: BinstBox) => d.reg_instances)
    .join("text")
    .attr("class", "soqlabel")
    .attr("text-anchor", (ri: RegInstance) => {
      switch (ri.side) {
        case THRU: return "middle";
        case LEFT: return "left";
        case RIGHT: return "end";
      }
    })
    .text((ri: RegInstance) => ri.label)
    .attr("x", (ri: RegInstance) => {
      switch (ri.side) {
        case THRU: return x_hat * (boxDrawProps.width / 2);
        case LEFT: return x_hat * (0 + boxDrawProps.marginLeft);
        case RIGHT: return x_hat * (boxDrawProps.width - boxDrawProps.marginRight);
      }
    })
    .attr("y", (d, i) => soq_y(i) * y_hat)
}

function add_soqcircles(g: d3.Selection<SVGGElement, BinstBox, SVGSVGElement, undefined>) {
  function get_soqcircle_points(binst: BinstBox) {
    return binst.reg_instances.flatMap((ri: RegInstance, i: number) => {
      switch (ri.side) {
        case LEFT: return { lr: 0, i: i };
        case RIGHT: return { lr: 1, i: i };
        case THRU: return [{ lr: 0, i: i }, { lr: 1, i: i }];
      }
    })
  }
  g.selectAll("circle")
    .data(get_soqcircle_points)
    .join("circle")
    .attr("class", "soqcircle")
    .attr("cx", d => (d.lr * boxDrawProps.width) * x_hat)
    .attr("cy", d => soq_y(d.i) * y_hat)
    .attr("r", boxDrawProps.soqCircleR)
}

function d3_join_binsts(binsts: Array<BinstBox>, tt: TransitionIn) {
  return svg.selectAll("g.binst")
    .data(binsts, (d: BinstBox) => d.i)
    .join(
      enter => enter.append("g")
        .attr("class", "binst")
        .attr("transform", d => `translate(${x(d.x)}, ${y(d.y)})`)
        .attr("opacity", 1.0)
        .call(g => g.append("rect")
          .attr("class", "binstbox")
          .attr("width", boxDrawProps.width * x_hat)
          .attr("height", d => (boxDrawProps.headerHeight + d.reg_instances.length * boxDrawProps.perSoqHeight) * y_hat)
          .attr("stroke", stroke_color)
          .lower()
        )
        .call(g => g.append("text")
          .attr("class", "binstlabel")
          .attr("fill", "red")
          .text(d => d.bloq.pretty_name + d.i)
          .attr("x", boxDrawProps.width / 2 * x_hat)
          .attr("y", boxDrawProps.marginTop)
        )
        .call(add_soqcircles)
        .call(add_soqlabels)
        .call(get_box_drag_behavior())
        .on("contextmenu", (event: PointerEvent, d: BinstBox) => { console.log(d); event.preventDefault() }),
      update => update
        .call(update => update.transition(tt)
          .attr("transform", d => `translate(${x(d.x)}, ${y(d.y)})`)
          .attr("opacity", 1.)
        ),
      exit => exit
        .call(exit => exit.transition(tt)
          .attr('transform', d => `translate(${x(d.x)}, ${y(d.y)}) scale(0)`)
          .attr('opacity', 0.)
          .remove()
        ),
    )
}



interface RefSoq {
  binst_i: number;
  soq_i: number;
}

type RefCxn = [RefSoq, RefSoq];

interface BloqResponse {
  binsts: Array<BinstBox>;
  cxns: Array<RefCxn>;
}


function resolve_ref_soq(ref_soq: RefSoq, binsts: Array<BinstBox>): Soquet {
  function find_binst(i: number): BinstBox {
    for (let j = 0; j < binsts.length; j++) {
      if (binsts[j].i === i) return binsts[j];
    }
    throw Error("Invalid binst_i");
  }

  return {
    binst: find_binst(ref_soq.binst_i),
    soq_i: ref_soq.soq_i,
  }
}

function resolve_ref_cxn(ref_cxn: RefCxn, binsts: Array<BinstBox>): Cxn {
  let [left, right] = ref_cxn;
  return [resolve_ref_soq(left, binsts), resolve_ref_soq(right, binsts)];
}

const driftobj = {
  drift: 0,
}

const simulation = d3.forceSimulation<BinstBox>()
  .force("col", d3.forceCollide(boxDrawProps.width/2))  /* TODO: this is centered on the corner */
  .force("repulse", d3.forceManyBody().strength(-50))
  .force("x", d3.forceX(d=>(d.gen_i+0.5+driftobj.drift)*(1.2*boxDrawProps.width)))
  .force("y", d3.forceY(500 - boxDrawProps.headerHeight - boxDrawProps.perSoqHeight).strength(0.01))
  /*.force("center", d3.forceCenter(500, 500))*/
  /*.force("link", d3.forceLink())*/
  ;

interface BinstLink{
  source: BinstBox;
  target: BinstBox;
}

function get_links(cxns:Array<Cxn>): Array<BinstLink>{
  const map = new Map();
  for (const [left, right] of cxns){
    map.set(`${left.binst.i}_${right.binst.i}`, {source:left.binst, target:right.binst});
  }
  return Array.from(map.values());

}

function handle_new_data(bloq_resp: BloqResponse) {
  const tt = svg.transition().duration(750);
  let binsts = bloq_resp.binsts;
  let ref_cxns = bloq_resp.cxns;
  console.log("Received", binsts.length, 'binsts')
  console.log("Received", ref_cxns.length, "connections")
  BOXES = binsts;
  CXNS = ref_cxns.map(c => resolve_ref_cxn(c, binsts));

  /*
  simulation.force<d3.ForceLink<BinstBox, BinstLink>>("link").links(get_links(CXNS)).distance(boxDrawProps.width*3);
  */
  let cxn_sel = d3_join_cxns(CXNS, tt);
  let binst_sel = d3_join_binsts(BOXES, tt);
  simulation.nodes(binsts);
  simulation.alpha(1);
  simulation.on("tick", () =>{
    driftobj.drift += 1;
    binst_sel.attr("transform", d => `translate(${x(d.x)}, ${y(d.y)})`)
    apply_cxn_pos_attrs(cxn_sel);

  })
}

function reset() {
  const tt = svg.transition().duration(750);
  BOXES = [];
  CXNS = [];
  d3_join_cxns(CXNS, tt);
  d3_join_binsts(BOXES, tt);
}

function expandBloq(bloq_key: string) {
  d3.json(bloq_key)
    .then(handle_new_data)
    .catch(reason => console.log("Error!", reason))
}

// Top level stuff

let ui_panel = d3.create("div").attr("id", "ui_panel");
let row0 = ui_panel.append("div");
row0.append("button").text("RESET").on("click", (event) => reset());
row0.append("button").text("Force").on("click", (event)=>simulation.alpha(1).restart());
let row1 = ui_panel.append("div");
row1.append("button").text("bloq/ModExp").on("click", (event) => expandBloq('bloq/ModExp'));
row1.append("button").text("bloq/ModExp/i0").on("click", (event) => expandBloq('bloq/ModExp/i0'));
row1.append("button").text("bloq/ModExp/i0/i4").on("click", (event) => expandBloq('bloq/ModExp/i0/i4'));
let row2 = ui_panel.append("div");
row2.append("button").text("bloq/TestParallelBloq").on("click", (event) => expandBloq('bloq/TestParallelBloq'));
row2.append("button").text("bloq/TestParallelBloq/i0").on("click", (event) => expandBloq('bloq/TestParallelBloq/i0'));

// Append the SVG element.
const container = document.createElement("div");
container.setAttribute("id", "container");
container.append(svg.node());
container.append(ui_panel.node());
document.body.appendChild(container);

d3.json('bloq/ModExp/i0').then(handle_new_data);
