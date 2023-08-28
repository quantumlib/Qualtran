
import * as d3 from "d3";

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

// Declare the y (vertical position) scale.
const y = d3.scaleLinear()
  .domain([0, 100])
  .range([marginTop, height - marginBottom]);

// Create the SVG container.
const svg = d3.create("svg")
  .attr("width", width)
  .attr("height", height);

const boxes = [
  { x: 5, y: 5 },
  { x: 50, y: 50 }
];

const boxwidth = 20;
const boxheight = 20;
const box_offsets = [
  { x: 0, y: 0 + 5 },
  { x: boxwidth, y: 0 + 5 },
  { x: 0, y: boxheight - 5 },
  { x: boxwidth, y: boxheight - 5 },
];

const cxns = [
  { lefti: 0, righti: 1, leftport: 3, rightport: 0 }
];


function my_box_drag() {

  function dragstarted(event, d) {
    console.log("drag start", d)
    d3.select(this).raise().attr("stroke", "red");
  }

  function dragged(event, d) {
    d.x = x.invert(event.x);
    d.y = y.invert(event.y);
    d3.select(this).attr("transform", `translate(${event.x}, ${event.y})`)
    update_cxns();
  }

  function dragended(event, d) {
    d3.select(this).attr("stroke", "white");
  }

  function subject(event, d) {
    console.log("subject", d)
    return { x: x(d.x), y: y(d.y) }
  }

  return d3.drag()
    .container(svg)
    .subject(subject)
    .on("start", dragstarted)
    .on("drag", dragged)
    .on("end", dragended);
}


function get_cxn_coords(cxn) {
  let box1 = boxes[cxn.lefti];
  let box2 = boxes[cxn.righti];
  let x1 = box1.x + box_offsets[cxn.leftport].x;
  let y1 = box1.y + box_offsets[cxn.leftport].y;
  let x2 = box2.x + box_offsets[cxn.rightport].x;
  let y2 = box2.y + box_offsets[cxn.rightport].y;
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

svg.selectAll("g.binst")
  .data(boxes)
  .join("g")
  .attr("class", "binst")
  .attr("stroke", "white")
  .attr("transform", d => `translate(${x(d.x)}, ${y(d.y)})`)
  .call(g => g.append("rect")
    .attr("width", x(boxwidth) - x(0)).attr("height", y(boxheight) - y(0))
  )
  .call(g => g.selectAll("circle")
    .data(box_offsets)
    .join("circle")
    .attr("fill", "white")
    .attr("cx", d => x(d.x) - x(0))
    .attr("cy", d => y(d.y) - y(0))
    .attr("r", 5)
  )
  .call(my_box_drag());


// Add the x-axis.
svg.append("g")
  .attr("transform", `translate(0,${height - marginBottom})`)
  .call(d3.axisBottom(x));

// Add the y-axis.
svg.append("g")
  .attr("transform", `translate(${marginLeft},0)`)
  .call(d3.axisLeft(y));

let ui_div = d3.create("div");

function expandBloq(event) {
  console.log(event)
}

ui_div.append("button").text("hey2").on("click", expandBloq);

// Append the SVG element.
container.append(svg.node());
container.append(ui_div.node());
