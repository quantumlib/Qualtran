
import * as d3 from "d3";

import './style.css';
import { canvas, get_scales, get_svg } from "./canvas";
import { BinstBox, Cxn, Soquet } from "./binst";

/* Set up drawing surface using function in the canvas module */
const { x, x_hat, y, y_hat } = get_scales(canvas);
const svg = get_svg(canvas, x, y)

const stroke_color = "black";
const highlight_stroke_color = "red";

const boxDrawProps = {
  width: 20,
  headerHeight: 6,
  perSoqHeight: 10,
  marginLeft: 2,
  marginRight: 2,
  soqCircleR: 6,
}

const boxes: Array<BinstBox> = [
  {
    x: 30, y: 15, title: "CNOT", soqs: [
      { thru: "ctrl" }, { thru: "ctrl2" }, { thru: "trg" }
    ]
  },
  {
    x: 60, y: 50, title: "Assym demo", soqs: [
      { left: "inx" }, { right: "outx" }
    ]
  },
  {
    x: 60, y: 5, title: "X", soqs: [
      { thru: "x" }
    ]
  },
  {
    x: 2, y: 30, title: "alloc", soqs: [
      { right: "x" }
    ]
  },
];

const cxns: Array<Cxn> = [
  [{ binst_i: 0, soq_i: 1 }, { binst_i: 1, soq_i: 0 }],
  [{ binst_i: 0, soq_i: 0 }, { binst_i: 2, soq_i: 0 }],
  [{ binst_i: 3, soq_i: 0 }, { binst_i: 0, soq_i: 2 }],
];

interface BinstBoxScreenCoords {
  x: number;
  y: number;
}

function get_box_drag_behavior(): d3.DragBehavior<SVGGElement, BinstBox, BinstBoxScreenCoords> {

  function dragstarted(event: any, d: BinstBox) {
    console.log("drag start", d)
    d3.select(this).raise().selectAll("rect").attr("stroke", highlight_stroke_color);
  }

  function dragged(event: any, d: BinstBox) {
    d.x = x.invert(event.x);
    d.y = y.invert(event.y);
    d3.select(this).attr("transform", `translate(${event.x}, ${event.y})`)
    update_cxns(cxns);
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

function get_cxn_coords(cxn: Cxn) {
  let [left, right] = cxn
  let box1 = boxes[left.binst_i];
  let box2 = boxes[right.binst_i];

  /* x1 and y1 are the "from" which is the right side of the box */
  let x1 = box1.x + boxDrawProps.width;
  let y1 = box1.y + soq_y(left.soq_i);

  /* x2 and y2 are the "to" which is the left side of the box */
  let x2 = box2.x + 0;
  let y2 = box2.y + soq_y(right.soq_i);

  return { x1: x(x1), y1: y(y1), x2: x(x2), y2: y(y2) }
}

function update_cxns(cxns: Array<Cxn>) {
  let cxn_data = cxns.map(get_cxn_coords)
  svg.selectAll("line.cxn")
    .data(cxn_data)
    .join("line")
    .attr("class", "cxn")
    .attr("stroke", stroke_color)
    .attr("stroke-width", 2)
    .attr("x1", d => d.x1).attr("y1", d => d.y1)
    .attr("x2", d => d.x2).attr("y2", d => d.y2)
}
update_cxns(cxns);


function soq_type(soq: Soquet) {
  if (soq.thru) return "thru";
  if (soq.left) return "left";
  if (soq.right) return "right";
  throw new Error("Bad soq type")
}

function add_soqlabels(g: d3.Selection<SVGGElement, BinstBox, SVGSVGElement, undefined>) {
  g.selectAll("text.soqlabel")
    .data((d: BinstBox) => d.soqs)
    .join("text")
    .attr("class", "soqlabel")
    .attr("text-anchor", (soq: Soquet) => {
      switch (soq_type(soq)) {
        case "thru": return "middle";
        case "left": return "left";
        case "right": return "end";
      }
    })
    .text((soq: Soquet) => {
      switch (soq_type(soq)) {
        case "thru": return soq.thru;
        case "left": return soq.left;
        case "right": return soq.right;
      }
    })
    .attr("x", (soq: Soquet) => {
      switch (soq_type(soq)) {
        case "thru": return x_hat * (boxDrawProps.width / 2);
        case "left": return x_hat * (0 + boxDrawProps.marginLeft);
        case "right": return x_hat * (boxDrawProps.width - boxDrawProps.marginRight);
      }
    })
    .attr("y", (d, i) => soq_y(i) * y_hat)
}

function add_soqcircles(g: d3.Selection<SVGGElement, BinstBox, SVGSVGElement, undefined>) {
  function get_soqcircle_points(box: BinstBox) {
    return box.soqs.flatMap((soq: Soquet, i: number) => {
      switch (soq_type(soq)) {
        case "left": return { lr: 0, i: i };
        case "right": return { lr: 1, i: i };
        case "thru": return [{ lr: 0, i: i }, { lr: 1, i: i }];
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


/* make binst boxes */
svg.selectAll("g.binst")
  .data(boxes)
  .join("g")
  .attr("class", "binst")
  .attr("transform", d => `translate(${x(d.x)}, ${y(d.y)})`)
  .call(g => g.append("rect")
    .attr("class", "binstbox")
    .attr("width", boxDrawProps.width * x_hat)
    .attr("height", d => (boxDrawProps.headerHeight + d.soqs.length * boxDrawProps.perSoqHeight) * y_hat)
    .attr("stroke", stroke_color)
  )
  .call(g => g.append("text")
    .attr("class", "binstlabel")
    .attr("fill", "red")
    .text(d => d.title)
    .attr("x", boxDrawProps.width / 2 * x_hat)
    .attr("y", boxDrawProps.headerHeight * 0.9 * y_hat)
  )
  .call(add_soqcircles)
  .call(add_soqlabels)
  .call(get_box_drag_behavior());



let ui_panel = d3.create("div").attr("id", "ui_panel");

function expandBloq(event: any) {
  console.log(event)
}

ui_panel.append("button").text("Decompose").on("click", expandBloq);

// Append the SVG element.
const container = document.createElement("div");
container.setAttribute("id", "container");
container.append(svg.node());
container.append(ui_panel.node());
document.body.appendChild(container);
