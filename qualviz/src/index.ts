
import * as d3 from "d3";

import './style.css';
import { canvas, get_scales, get_svg, stroke_color, highlight_stroke_color, boxDrawProps } from "./canvas";
import { Binst, Cxn, RegInstance, Bloq } from "./binst";

/* Set up drawing surface using function in the canvas module */
const { x, x_hat, y, y_hat } = get_scales(canvas);
const svg = get_svg(canvas, x, y)

let RIS: Array<RegInstance> = [];
let CXNS: Array<Cxn> = [];

interface BinstBoxScreenCoords {
  x: number;
  y: number;
}

type TransitionIn = d3.Transition<d3.BaseType, any, any, any>;


function get_ri_drag_behavior(): d3.DragBehavior<SVGGElement, RegInstance, BinstBoxScreenCoords> {

  function dragstarted(event: any, d: RegInstance) {
    d3.select(this).raise().selectAll("circle").attr("stroke", highlight_stroke_color);
    if (!event.active) simulation.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
  }

  function dragged(event: any, d: RegInstance) {
    d.fx = x.invert(event.x);
    d.fy = y.invert(event.y);
  }

  function dragended(event: any, d: RegInstance) {
    d3.select(this).selectAll("circle").attr("stroke", stroke_color);
    if (!event.active) simulation.alphaTarget(0);
    d.fx = null;
    d.fy = null;
  }

  function subject(event: any, d: RegInstance): BinstBoxScreenCoords {
    return { x: x(d.x), y: y(d.y) }
  }

  return d3.drag<SVGGElement, RegInstance, BinstBoxScreenCoords>()
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
  return `${left.binst.i}_${left.i}_${right.binst.i}_${right.i}`
}

function apply_cxn_pos_attrs(sel: any) {
  return sel
    /* x1 and y1 are the "from" which is the right side of the box */
    .attr("x1", (d: Cxn) => x(d[0].x))
    .attr("y1", (d: Cxn) => y(d[0].y))
    /* x2 and y2 are the "to" which is the left side of the box */
    .attr("x2", (d: Cxn) => x(d[1].x))
    .attr("y2", (d: Cxn) => y(d[1].y))
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

function dirty(d: Binst) {
  const ris = d.reg_instances.sort();
  const x0 = ris.reduce((m: number, ri: RegInstance) => Math.min(m, ri.x - boxDrawProps.width / 3), Infinity);
  const x1 = ris.reduce((m: number, ri: RegInstance) => Math.max(m, ri.x + boxDrawProps.width / 3), -Infinity);
  const y0 = ris.reduce((m: number, ri: RegInstance) => Math.min(m, ri.y - boxDrawProps.width / 3), Infinity);
  const y1 = ris.reduce((m: number, ri: RegInstance) => Math.max(m, ri.y + boxDrawProps.width / 3), -Infinity);

  d.x0 = x0;
  d.x1 = x1;
  d.y0 = y0;
  d.y1 = y1;

  return {
    x0: x0,
    x1: x1,
    y0: y0,
    y1: y1,
  }
}

function d3_join_binsts(binsts: Array<Binst>) {
  svg.selectAll("g.binstgroup")
    .data(binsts, (d: Binst) => d.i)
    .join(
      enter => enter.append("g")
        .attr("class", "binstgroup")
        .each(dirty)
        .call(g => {
          g.append("rect")
            .attr("x", d => x(d.x0))
            .attr("width", d => x(d.x1) - x(d.x0))
            .attr("y", d => y(d.y0))
            .attr("height", d => y(d.y1) - y(d.y0))
            .attr("fill", "none")
            .attr("stroke", "grey");
        }),
      update => update
        .each(dirty)
        .call(uu => uu.selectAll("rect")
          .attr("x", (d: Binst) => x(d.x0))
          .attr("width", (d: Binst) => x(d.x1) - x(d.x0))
          .attr("y", (d: Binst) => y(d.y0))
          .attr("height", (d: Binst) => y(d.y1) - y(d.y0)),
        ),
      exit => exit.remove()
    )
}


const THRU = "Side.THRU";
const LEFT = "Side.LEFT";
const RIGHT = "Side.RIGHT";



function d3_join_ris(ris: Array<RegInstance>, tt: TransitionIn) {
  return svg.selectAll("g.ri")
    .data(ris, (d: RegInstance) => `${d.binst.i}_${d.i}`)
    .join(
      enter => enter.append("g")
        .attr("class", "ri")
        .attr("transform", d => `translate(${x(d.x)}, ${y(d.y)})`)
        .attr("opacity", 1.0)
        .call(g => g.append("circle")
          .attr("class", "ricircle")
          .attr("r", boxDrawProps.width / 3)
          .attr("cx", 0).attr("cy", 0)
          .attr("stroke", stroke_color)
          .attr("fill", "white")
          .lower()
        )
        .call(g => g.append("text")
          .attr("class", "rilabel")
          .attr("fill", "black")
          .text(d => d.binst.bloq.pretty_name + d.binst.i + ' ' + d.label + d.i)
          .attr("x", 0)
          .attr("y", 0)
        )
        .call(get_ri_drag_behavior()),
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



function resolve_ref_soq(ref_soq: RefSoq, ris: Array<RegInstance>): RegInstance {
  function find_ri(binst_i: number, ri_i: number): RegInstance {
    for (let j = 0; j < ris.length; j++) {
      if (ris[j].binst.i == binst_i && ris[j].i === ri_i) return ris[j];
    }
    throw Error("Invalid binst_i");
  }
  return find_ri(ref_soq.binst_i, ref_soq.soq_i);
}

function resolve_ref_cxn(ref_cxn: RefCxn, ris: Array<RegInstance>): Cxn {
  let [left, right] = ref_cxn;
  return [resolve_ref_soq(left, ris), resolve_ref_soq(right, ris)];
}

const driftobj = {
  drift: 0,
}

const simulation = d3.forceSimulation<RegInstance>()
  .force("col", d3.forceCollide(boxDrawProps.width / 3))  /* TODO: this is centered on the corner */
  .force("repulse", d3.forceManyBody().strength(-50))
  .force("x", d3.forceX(d => (d.binst.gen_i + 1) * (100)))
  .force("y", d3.forceY(300).strength(0.01))
  /*.force("center", d3.forceCenter(500, 500))*/
  /*.force("link", d3.forceLink())*/
  ;

interface BinstLink {
  source: Binst;
  target: Binst;
}

function get_links(cxns: Array<Cxn>): Array<BinstLink> {
  const map = new Map();
  for (const [left, right] of cxns) {
    map.set(`${left.binst.i}_${right.binst.i}`, { source: left.binst, target: right.binst });
  }
  return Array.from(map.values());

}

interface RefRegInst {
  label: string;
  side: string;
}

interface RefBinst {
  i: number;
  x: number;
  y: number;
  gen_i: number;
  bloq: Bloq;
  reg_instances: Array<RefRegInst>;
}

interface BloqResponse {
  binsts: Array<RefBinst>;
  cxns: Array<RefCxn>;
}

function handle_new_data(bloq_resp: BloqResponse) {
  const tt = svg.transition().duration(750);
  simulation.stop();
  let ref_binsts = bloq_resp.binsts;
  let ref_cxns = bloq_resp.cxns;
  console.log("Received", ref_binsts.length, 'binsts')
  console.log("Received", ref_cxns.length, "connections")

  const ris: Array<RegInstance> = [];
  const binstmap = new Map();
  for (const refbinst of ref_binsts) {
    for (let i = 0; i < refbinst.reg_instances.length; i++) {
      const rri = refbinst.reg_instances[i];
      let binst = binstmap.get(refbinst.i);
      if (!binst) {
        binst = { i: refbinst.i, bloq: refbinst.bloq, gen_i: refbinst.gen_i, reg_instances: [] }
        binstmap.set(refbinst.i, binst)
      }

      ris.push({
        i: i,
        label: rri.label,
        side: rri.side,
        x: refbinst.x,
        y: refbinst.y + 10 * i,
        binst: binst
      });
    }
  }
  for (const ri of ris) {
    ri.binst.reg_instances.push(ri);
  }
  const binsts: Array<Binst> = Array.from(binstmap.values());

  RIS = ris;
  CXNS = ref_cxns.map(c => resolve_ref_cxn(c, ris));

  /*
  simulation.force<d3.ForceLink<BinstBox, BinstLink>>("link").links(get_links(CXNS)).distance(boxDrawProps.width*3);
  */
  let cxn_sel = d3_join_cxns(CXNS, tt);
  let ri_sel = d3_join_ris(RIS, tt);
  let binst_sel = d3_join_binsts(binsts);
  simulation.nodes(RIS).alpha(1).restart();
  simulation.on("tick", () => {
    ri_sel.attr("transform", d => `translate(${x(d.x)}, ${y(d.y)})`)
    apply_cxn_pos_attrs(cxn_sel);
    d3_join_binsts(binsts);
  })
}

function reset() {
  const tt = svg.transition().duration(750);
  simulation.stop().on("tick", () => null).nodes([]);
  RIS = [];
  CXNS = [];
  d3_join_cxns([], tt);
  d3_join_ris([], tt);
  d3_join_binsts([]);
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
row0.append("button").text("Force").on("click", (event) => simulation.alpha(1).restart());
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
