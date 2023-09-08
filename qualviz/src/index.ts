
import * as d3 from "d3";

import './style.css';
import { canvas, get_scales, get_svg, stroke_color, highlight_stroke_color, boxDrawProps } from "./canvas";
import { BinstBox, Cxn, Soquet } from "./binst";

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
    console.log("drag start", d)
    d3.select(this).raise().selectAll("rect").attr("stroke", highlight_stroke_color);
  }

  function dragged(event: any, d: BinstBox) {
    d.x = x.invert(event.x);
    d.y = y.invert(event.y);
    d3.select(this).attr("transform", `translate(${event.x}, ${event.y})`)
    update_cxns(CXNS, BOXES);
  }

  function dragended(event: any, d: BinstBox) {
    d3.select(this).selectAll("rect").attr("stroke", stroke_color);
  }

  function subject(event: any, d: BinstBox): BinstBoxScreenCoords {
    console.log("subject", d)
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

function get_cxn_coords(cxn: Cxn, binsts: Array<BinstBox>) {
  let [left, right] = cxn

  function find_binst(i: number): BinstBox {
    for (let j = 0; j < binsts.length; j++) {
      if (binsts[j].i === i) return binsts[j];
    }
    throw Error("Invalid binst_i");
  }

  let binst1 = find_binst(left.binst_i);
  let binst2 = find_binst(right.binst_i);

  /* x1 and y1 are the "from" which is the right side of the box */
  let x1 = binst1.x + boxDrawProps.width;
  let y1 = binst1.y + soq_y(left.soq_i);

  /* x2 and y2 are the "to" which is the left side of the box */
  let x2 = binst2.x + 0;
  let y2 = binst2.y + soq_y(right.soq_i);

  return { x1: x(x1), y1: y(y1), x2: x(x2), y2: y(y2), key: `${left.binst_i}_${left.soq_i}_${right.binst_i}_${right.soq_i}` }
}

interface HasKey {
  key: string;
}

function d3_join_cxns(cxns: Array<Cxn>, binsts: Array<BinstBox>, tt: TransitionIn) {
  let cxn_data = cxns.map(cxn => get_cxn_coords(cxn, binsts))
  svg.selectAll("line.cxn")
    .data(cxn_data, (d: HasKey) => d.key)
    .join(
      enter => enter.append("line")
        .attr("class", "cxn")
        .attr("stroke", stroke_color)
        .attr("stroke-width", 2)
        .attr("opacity", 0.0)
        .attr("x1", d => d.x1).attr("y1", d => d.y1)
        .attr("x2", d => d.x2).attr("y2", d => d.y2)
        .call(enter => enter.transition(tt)
          .attr("opacity", 1.0)
        ),
      update => update
        .call(update => update.transition(tt)
          .attr("x1", d => d.x1)
          .attr("y1", d => d.y1)
          .attr("x2", d => d.x2)
          .attr("y2", d => d.y2)
          .attr("opacity", 1.0)
        ),
      exit => exit
        .call(exit => exit.transition(tt)
          .attr("opacity", 0.0)
          .remove()
        ),
    )
}

function update_cxns(cxns: Array<Cxn>, binsts: Array<BinstBox>) {
  let cxn_data = cxns.map(cxn => get_cxn_coords(cxn, binsts))
  svg.selectAll("line.cxn")
    .data(cxn_data, (d: HasKey) => d.key)
    .join("line")
    .attr("class", "cxn")
    .attr("stroke", stroke_color)
    .attr("stroke-width", 2)
    .attr("x1", d => d.x1).attr("y1", d => d.y1)
    .attr("x2", d => d.x2).attr("y2", d => d.y2)
}

const THRU = "Side.THRU";
const LEFT = "Side.LEFT";
const RIGHT = "Side.RIGHT";


function add_soqlabels(g: d3.Selection<SVGGElement, BinstBox, SVGSVGElement, undefined>) {
  g.selectAll("text.soqlabel")
    .data((d: BinstBox) => d.soqs)
    .join("text")
    .attr("class", "soqlabel")
    .attr("text-anchor", (soq: Soquet) => {
      switch (soq.side) {
        case THRU: return "middle";
        case LEFT: return "left";
        case RIGHT: return "end";
      }
    })
    .text((soq: Soquet) => soq.label)
    .attr("x", (soq: Soquet) => {
      switch (soq.side) {
        case THRU: return x_hat * (boxDrawProps.width / 2);
        case LEFT: return x_hat * (0 + boxDrawProps.marginLeft);
        case RIGHT: return x_hat * (boxDrawProps.width - boxDrawProps.marginRight);
      }
    })
    .attr("y", (d, i) => soq_y(i) * y_hat)
}

function add_soqcircles(g: d3.Selection<SVGGElement, BinstBox, SVGSVGElement, undefined>) {
  function get_soqcircle_points(box: BinstBox) {
    return box.soqs.flatMap((soq: Soquet, i: number) => {
      switch (soq.side) {
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
  svg.selectAll("g.binst")
    .data(binsts, (d: BinstBox) => d.i)
    .join(
      enter => enter.append("g")
        .attr("class", "binst")
        .attr("transform", d => `translate(${x(d.x)}, ${y(d.y)})`)
        .attr("opacity", 1.0)
        .call(g => g.append("rect")
          .attr("class", "binstbox")
          .attr("width", boxDrawProps.width * x_hat)
          .attr("height", d => (boxDrawProps.headerHeight + d.soqs.length * boxDrawProps.perSoqHeight) * y_hat)
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


interface BloqResponse {
  binsts: Array<BinstBox>;
  cxns: Array<Cxn>;
}

function handle_new_data(bloq_resp: BloqResponse) {
  const tt = svg.transition().duration(750);
  let binsts = bloq_resp.binsts;
  let cxns = bloq_resp.cxns;
  console.log("Received", binsts.length, 'binsts')
  console.log("Received", cxns.length, "connections")
  console.log(binsts)
  console.log(cxns)
  BOXES = binsts;
  CXNS = cxns;
  d3_join_cxns(CXNS, BOXES, tt);
  d3_join_binsts(BOXES, tt);
}

function reset() {
  const tt = svg.transition().duration(750);
  BOXES = [];
  CXNS = [];
  d3_join_cxns(CXNS, BOXES, tt);
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

d3.json('bloq/ModExp').then(handle_new_data);
