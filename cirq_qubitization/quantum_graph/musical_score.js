const r = 3;
const xType = d3.scaleLinear; // type of x-scale
const yType = d3.scaleLinear; // type of y-scale
const marginTop = 20; // top margin, in pixels
const marginRight = 30; // right margin, in pixels
const marginBottom = 30; // bottom margin, in pixels
const marginLeft = 40; // left margin, in pixels
const inset = r * 2; // inset the default range, in pixels
const insetTop = inset; // inset the default y-range
const insetRight = inset; // inset the default x-range
const insetBottom = inset; // inset the default y-range
const insetLeft = inset; // inset the default x-range

const halfHeight = 14;
const halfWidth = 22;

const n_x = xDomain[1] - xDomain[0];
const n_y = yDomain[1] - yDomain[0];
const width = n_x * (halfWidth * 2.5) + marginLeft + marginRight + insetLeft + insetRight;
const height = n_y * (3 * halfHeight) + marginBottom + insetBottom + marginTop + insetTop;
const xRange = [marginLeft + insetLeft, width - marginRight - insetRight];
const yRange = [height - marginBottom - insetBottom, marginTop + insetTop];

// Construct scales and axes.
const xScale = xType(xDomain, xRange);
const yScale = yType([yDomain[1], yDomain[0]], yRange);
const xAxis = d3.axisBottom(xScale).ticks(width / 80);
const yAxis = d3.axisLeft(yScale).ticks(height / 50);


function makeCanvas(sel) {
  const svg = sel.append("svg")
    .attr("width", width)
    .attr("height", height)
    .attr("viewBox", [0, 0, width, height])
    .attr("style", "max-width: 100%; height: auto; height: intrinsic;");

  svg.append("g")
    .attr("transform", `translate(0,${height - marginBottom})`)
    .call(xAxis)
    .call(g => g.select(".domain").remove())

  svg.append("g")
    .attr("transform", `translate(${marginLeft},0)`)
    .call(yAxis)
    .call(g => g.select(".domain").remove())

  const datag = svg.append("g")
    .attr("id", "datag");

  return [svg, datag];
}

const [canvas, DATA_G] = makeCanvas(d3.select("#content"));

function drawVlines(vlines, tt) {
  DATA_G.selectAll('line.v')
    .data(vlines, d => d.x)
    .join(
      enter => enter.append("line")
        .attr("class", "v")
        .attr("stroke", "black")
        .attr('x1', d => xScale(d.x))
        .attr('x2', d => xScale(d.x))
        .attr('y1', d => yScale(d.bottom_y))
        .attr('y2', d => yScale(d.bottom_y))
        .call(enter => enter.transition(tt)
          .attr('y2', d => yScale(d.top_y))
        )
        .lower(),
      update => update
        .call(update => update.transition(tt)
          .attr('x1', d => xScale(d.x))
          .attr('x2', d => xScale(d.x))
          .attr('y1', d => yScale(d.bottom_y))
          .attr('y2', d => yScale(d.top_y))
        )
        .lower(),
      exit => exit
        .call(exit => exit.transition(tt)
          .attr('y2', d => yScale(d.bottom_y))
          .remove()
        ),
    );

}

function drawHlines(hlines, tt) {
  DATA_G.selectAll('line.h')
    .data(hlines, d => [d.y, d.seq_x_start])
    .join(
      enter => enter.append("line")
        .attr("class", "h")
        .attr("stroke", "lightblue")
        .attr('x1', d => xScale(d.seq_x_start))
        .attr('x2', d => xScale(d.seq_x_start))
        .attr('y1', d => yScale(d.y))
        .attr('y2', d => yScale(d.y))
        .call(enter => enter.transition(tt)
          .attr('x2', d => xScale(d.seq_x_end))
        )
        .lower(),
      update => update
        .call(update => update.transition(tt)
          .attr('x1', d => xScale(d.seq_x_start))
          .attr('x2', d => xScale(d.seq_x_end))
          .attr('y1', d => yScale(d.y))
          .attr('y2', d => yScale(d.y))
        )
        .lower(),
      exit => exit
        .call(exit => exit.transition(tt)
          .attr('x2', d => xScale(d.seq_x_start))
          .remove()
        ),
    );
}

function drawCircles(data, x, y, tt) {
  DATA_G.selectAll("circle.circle")
    .data(data.filter(d => d.symb_cls === "Circle"), d => d.ident)
    .join(
      enter => enter.append("circle")
        .attr("class", "circle")
        .attr("stroke", "black")
        .attr("r", 5)
        .attr("fill", d => d.symb_attributes.filled ? "black" : "white")
        .attr("cx", d => xScale(x(d)))
        .attr("cy", 0)
        .attr("opacity", 0)
        .call(enter => enter.transition(tt)
          .attr("cy", d => yScale(y(d)))
          .attr("opacity", 1.0)
        ),
      update => update
        .attr("fill", d => d.symb_attributes.filled ? "black" : "white")
        .call(enter => enter.transition(tt)
          .attr("cy", d => yScale(y(d)))
          .attr("cx", d => xScale(x(d)))
          .attr("opacity", 1.0)
        ),
      exit => exit.transition(tt)
        .attr("cy", 0)
        .attr("opacity", 0)
        .remove()
    );
}

function drawModPlus(data, x, y, tt) {
  DATA_G.selectAll("g.modplus")
    .data(data.filter(d => d.symb_cls == "ModPlus"), d => d.ident)
    .join(
      enter => enter.append("g")
        .attr("class", "modplus")
        .attr("stroke", "black")
        .attr("transform", (d, i) => `translate(${xScale(x(d))}, ${yScale(yDomain[1])})`)
        .attr("opacity", 0.)
        .call(g => g.append("path")
          .attr("d", "M 0 -7 l 0 14 M -7 0 l 14 0")
        )
        .call(g => g.append("circle")
          .attr("fill", "none")
          .attr("r", 7)
        )
        .call(g => g.transition(tt)
          .attr("transform", (d, i) => `translate(${xScale(x(d))}, ${yScale(y(d))})`)
          .attr("opacity", 1.)
        ),
      update => update.transition(tt)
        .attr("transform", (d, i) => `translate(${xScale(x(d))}, ${yScale(y(d))})`)
        .attr("opacity", 1.),
      exit => exit.transition(tt)
        .attr("transform", (d, i) => `translate(${xScale(x(d))}, ${yScale(yDomain[1])})`)
        .attr("opacity", 0.)
        .remove()
    );
}

function drawAnyTextBox(data, x, y, tt, clsname, drawbox) {
  DATA_G.selectAll(`g.${clsname}`)
    .data(data.filter(d => d.symb_cls === clsname), d => d.ident)
    .join(
      enter => enter
        .append("g")
        .attr("class", clsname)
        .attr("transform", (d, i) => `translate(${xScale(x(d))}, ${yScale(y(d))})`)
        .on("click", (e, d) => console.log(e, d))
        .attr("opacity", 0.)
        .call(r => r.transition(tt)
          .attr("opacity", 1.)
        )
        .call(drawbox)
        .call(enter => enter.append("text")
          .attr("text-anchor", "middle")
          .attr("dominant-baseline", "middle")
          .text(d => d.symb_attributes.text)
        ),
      update => update
        .call(update => update
          .select("text")
          .text(d => d.symb_attributes.text)
        )
        .call(update => update.transition(tt)
          .attr("transform", (d, i) => `translate(${xScale(x(d))}, ${yScale(y(d))})`)
          .attr("opacity", 1.)
        ),
      exit => exit.transition(tt)
        .attr("opacity", 0.0)
        .remove()
    )
}

function drawTextBox(data, x, y, tt) {
  drawAnyTextBox(data, x, y, tt, "TextBox",
    enter => enter.append("rect")
      .attr("fill", "white").attr("stroke", "black")
      .attr("width", 2 * halfWidth).attr("height", 2 * halfHeight)
      .attr("y", -halfHeight).attr("x", -halfWidth)
  )
}

function drawRarrowTextBox(data, x, y, tt) {
  drawAnyTextBox(data, x, y, tt, "RarrowTextBox",
    g => g.append("polygon")
      .attr("fill", "white").attr("stroke", "black")
      .attr("points", `-${halfWidth},-${halfHeight} 15,-${halfHeight} ${halfWidth},0 15,${halfHeight} -${halfWidth},${halfHeight}`)
      .attr("y", -halfHeight)
      .attr("x", -halfWidth)
  )
}

function drawLarrowTextBox(data, x, y, tt) {
  drawAnyTextBox(data, x, y, tt, "LarrowTextBox",
    g => g.append("polygon")
      .attr("fill", "white").attr("stroke", "black")
      .attr("points", `-${halfWidth},0 -15,-${halfHeight} ${halfWidth},-${halfHeight} ${halfWidth},${halfHeight} -15,${halfHeight}`)
      .attr("y", -halfHeight)
      .attr("x", -halfWidth)
  )
}

function drawText(data, x, y, tt) {
  drawAnyTextBox(data, x, y, tt, "Text", g => g)
}

function musicalScore(data, vlines, hlines, { x, y }) {
  const tt = DATA_G.transition().duration(750);
  drawVlines(vlines, tt);
  drawHlines(hlines, tt)
  drawCircles(data, x, y, tt);
  drawModPlus(data, x, y, tt);
  drawTextBox(data, x, y, tt);
  drawLarrowTextBox(data, x, y, tt);
  drawRarrowTextBox(data, x, y, tt);
  drawText(data, x, y, tt);
}

function showError(message, details) {
  d3.select("#content").append("h3").attr("style", "color:red").text(message)
  d3.select("#content").append("p").text(details)
}

function make_from_data(fn) {
  d3.json(fn)
    .then(data => musicalScore(data.soqs, data.vlines, data.hlines, { x: (d) => d.seq_x, y: (d) => d.y }))
    .catch(error => showError(`Error loading '${fn}': ${error}`, "Make sure the file exists and make sure you're serving this page from `python -m http.server` instead of from a file"))
}

