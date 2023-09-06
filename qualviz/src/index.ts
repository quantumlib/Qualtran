
import * as d3 from "d3";

import './style.css';

// Declare the chart dimensions and margins.
const width = 800;
const height = 600;
const marginTop = 20;
const marginRight = 20;
const marginBottom = 30;
const marginLeft = 40;

// Declare the x (horizontal position) scale.
const x = d3.scaleLinear()
  .domain([0, 100])
  .range([marginLeft, width - marginRight]);
const x_hat = x(1) - x(0);

// Declare the y (vertical position) scale.
const y = d3.scaleLinear()
  .domain([0, 100])
  .range([marginTop, height - marginBottom]);
const y_hat = y(1) - y(0);

// Create the SVG container.
const svg = d3.create("svg")
  .attr("width", width)
  .attr("height", height);

interface Soquet {
  thru?: string;
  left?: string;
  right?: string;

}

interface BinstBox {
  x: number;
  y: number;
  title: string;
  soqs: Array<Soquet>;
}

const boxes: Array<BinstBox> = [
  {
    x: 5, y: 5, title: "CNOT", soqs: [
      { thru: "ctrl" }, {thru: "ctrl2"}, { thru: "trg" }
    ]
  },
  {
    x: 50, y: 50,  title: "Assym demo", soqs: [
      { left: "inx" }, { right: "outx" }
    ]
  }
];

const boxWidth = 20;
const boxHeaderHeight = 6;
const perSoqHeight = 10;

interface Port {
  binst_i: number;
  soq_i: number;
}

type Cxn = [Port, Port];


const cxns: Array<Cxn> = [
  [{binst_i: 0, soq_i:1}, {binst_i: 1, soq_i: 0}]
];

interface BinstBoxScreenCoords {
  x: number;
  y: number;
}


function get_box_drag_behavior(): d3.DragBehavior<SVGGElement, BinstBox, BinstBoxScreenCoords> {

  function dragstarted(event: any, d: BinstBox) {
    console.log("drag start", d)
    d3.select(this).raise().selectAll("rect").attr("stroke", "red");
  }

  function dragged(event: any, d: BinstBox) {
    d.x = x.invert(event.x);
    d.y = y.invert(event.y);
    d3.select(this).attr("transform", `translate(${event.x}, ${event.y})`)
    update_cxns();
  }

  function dragended(event: any, d: BinstBox) {
    d3.select(this).selectAll("rect").attr("stroke", "white");
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


function get_cxn_coords(cxn: Cxn) {
  let [left, right] = cxn
  let box1 = boxes[left.binst_i];
  let box2 = boxes[right.binst_i];

  /* x1 and y1 are the "from" which is the right side of the box */
  let x1 = box1.x + boxWidth;
  let y1 = box1.y + boxHeaderHeight + (perSoqHeight * left.soq_i); /* stacking left and right complicates */

  /* x2 and y2 are the "to" which is the left side of the box */
  let x2 = box2.x + 0;
  let y2 = box2.y + boxHeaderHeight + (perSoqHeight * right.soq_i);
  return { x1: x(x1), y1: y(y1), x2: x(x2), y2: y(y2) }
}

function update_cxns() {
  let cxn_data = cxns.map(get_cxn_coords)
  svg.selectAll("line.cxn")
    .data(cxn_data)
    .join("line")
    .attr("class", "cxn")
    .attr("stroke", "green")
    .attr("stroke-width", 2)
    .attr("x1", d => d.x1).attr("y1", d => d.y1)
    .attr("x2", d => d.x2).attr("y2", d => d.y2)
}
update_cxns();


function get_soquet_text(soq: Soquet): string {
  return soq.thru? soq.thru : `${soq.left} | ${soq.right}`
}


/* make binst boxes */
svg.selectAll("g.binst")
  .data(boxes)
  .join("g")
  .attr("class", "binst")
  .attr("transform", d => `translate(${x(d.x)}, ${y(d.y)})`)
  .call(g => g.append("rect")
    .attr("width", boxWidth*x_hat)
    .attr("height", d=>(boxHeaderHeight + d.soqs.length*perSoqHeight)*y_hat)
    .attr("stroke", "white")
  )
  .call(g=> g.append("text")
    .attr("class", "binstlabel")
    .attr("fill", "red")
    .text(d=> d.title)
    .attr("x", boxWidth/2*x_hat)
    .attr("y", boxHeaderHeight*0.9*y_hat)
  )
  .call(g => g.selectAll("circle")
    .data(d=>d.soqs)
    .join("circle")
    .attr("fill", "white")
    .attr("cx", d => 0)
    .attr("cy", (d,i) => (boxHeaderHeight + perSoqHeight*(i+0.5)) * y_hat)
    .attr("r", 5)
  )
  .call(g=> g.selectAll("text.soqlabel")
    .data((d:BinstBox)=>d.soqs)
    .join("text")
    .attr("class", "soqlabel")
    .text(get_soquet_text)
    .attr("x", (d, i) => 10)
    .attr("y", (d, i) => (boxHeaderHeight + perSoqHeight*(i+0.5))*y_hat)
    .attr("fill", "white")
  )
  .call(get_box_drag_behavior());


// Add the x-axis.
svg.append("g")
  .attr("transform", `translate(0,${height - marginBottom})`)
  .call(d3.axisBottom(x));

// Add the y-axis.
svg.append("g")
  .attr("transform", `translate(${marginLeft},0)`)
  .call(d3.axisLeft(y));

let ui_div = d3.create("div");

function expandBloq(event: any) {
  console.log(event)
}

ui_div.append("button").text("hey3").on("click", expandBloq);

// Append the SVG element.
const container = document.createElement("div");
container.setAttribute("id", "container");
container.append(svg.node());
container.append(ui_div.node());
document.body.appendChild(container);
