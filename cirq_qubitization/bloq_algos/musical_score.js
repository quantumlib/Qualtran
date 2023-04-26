

// Copyright 2021 Observable, Inc.
// Released under the ISC license.
// https://observablehq.com/@d3/scatterplot
function Scatterplot(data, vlines, {
  x = ([x]) => x, // given d in data, returns the (quantitative) x-value
  y = ([, y]) => y, // given d in data, returns the (quantitative) y-value
  r = 3, // (fixed) radius of dots, in pixels
  title, // given d in data, returns the title
  marginTop = 20, // top margin, in pixels
  marginRight = 30, // right margin, in pixels
  marginBottom = 30, // bottom margin, in pixels
  marginLeft = 40, // left margin, in pixels
  inset = r * 2, // inset the default range, in pixels
  insetTop = inset, // inset the default y-range
  insetRight = inset, // inset the default x-range
  insetBottom = inset, // inset the default y-range
  insetLeft = inset, // inset the default x-range
  xType = d3.scaleLinear, // type of x-scale
  xDomain, // [xmin, xmax]
  yType = d3.scaleLinear, // type of y-scale
  yDomain, // [ymin, ymax]
  xLabel, // a label for the x-axis
  yLabel, // a label for the y-axis
  xFormat, // a format specifier string for the x-axis
  yFormat, // a format specifier string for the y-axis
  fill = "none", // fill color for dots
  stroke = "black", // stroke color for the dots
  strokeWidth = 1.5, // stroke width for dots
  halo = "#fff", // color of label halo
  haloWidth = 3 // padding around the labels
} = {}) {
  // Compute values.
  const X = d3.map(data, x);
  const Y = d3.map(data, y);
  const T = title == null ? null : d3.map(data, title);
  const I = d3.range(X.length).filter(i => !isNaN(X[i]) && !isNaN(Y[i]));

  // Compute default domains.
  if (xDomain === undefined) xDomain = d3.extent(X);
  if (yDomain === undefined) yDomain = d3.extent(Y);
  const width = d3.max(X) * 50; // mph hack
  const height = d3.max(Y) * 50; // mph hack
  const xRange = [marginLeft + insetLeft, width - marginRight - insetRight];
  const yRange = [height - marginBottom - insetBottom, marginTop + insetTop];

  // Construct scales and axes.
  const xScale = xType(xDomain, xRange);
  const yScale = yType(yDomain, yRange);
  const xAxis = d3.axisBottom(xScale).ticks(width / 80, xFormat);
  const yAxis = d3.axisLeft(yScale).ticks(height / 50, yFormat);

  const svg = d3.create("svg")
      .attr("width", width)
      .attr("height", height)
      .attr("viewBox", [0, 0, width, height])
      .attr("style", "max-width: 100%; height: auto; height: intrinsic;");

  svg.append("g")
      .attr("transform", `translate(0,${height - marginBottom})`)
      .call(xAxis)
      .call(g => g.select(".domain").remove())
      /*.call(g => g.selectAll(".tick line").clone()
          .attr("y2", marginTop + marginBottom - height)
          .attr("stroke-opacity", 0.1))*/
      .call(g => g.append("text")
          .attr("x", width)
          .attr("y", marginBottom - 4)
          .attr("fill", "currentColor")
          .attr("text-anchor", "end")
          .text(xLabel));

  svg.append("g")
      .attr("transform", `translate(${marginLeft},0)`)
      .call(yAxis)
      .call(g => g.select(".domain").remove())
      .call(g => g.selectAll(".tick line").clone()
          .attr("x2", width - marginLeft - marginRight)
          .attr("stroke-opacity", 0.1))
      .call(g => g.append("text")
          .attr("x", -marginLeft)
          .attr("y", 10)
          .attr("fill", "currentColor")
          .attr("text-anchor", "start")
          .text(yLabel));

  if (T) svg.append("g")
      .attr("font-family", "sans-serif")
      .attr("font-size", 10)
      .attr("stroke-linejoin", "round")
      .attr("stroke-linecap", "round")
    .selectAll("text")
    .data(I)
    .join("text")
      .attr("dx", 7)
      .attr("dy", "0.35em")
      .attr("x", i => xScale(X[i]))
      .attr("y", i => yScale(Y[i]))
      .text(i => T[i])
      .call(text => text.clone(true))
      .attr("fill", "none")
      .attr("stroke", halo)
      .attr("stroke-width", haloWidth);

  /************************
   * vlines
   ******************/
  svg.append('g')
    .attr("stroke", "black")
    .selectAll('line')
    .data(vlines)
    .join('line')
    .attr('x1', d=>xScale(d.x))
    .attr('x2', d=>xScale(d.x))
    .attr('y1', d=>yScale(d.bot_y))
    .attr('y2', d=>yScale(d.top_y))


  /************************
   * circles
   ******************/
  svg.append("g")
      .attr("fill", fill)
      .attr("stroke", stroke)
      .attr("stroke-width", strokeWidth)
    .selectAll("circle")
    .data(data.filter(d=>d.symb==="Circle"))
    .join("circle")
      .attr("cx", d => xScale(x(d)))
      .attr("cy", d => yScale(y(d)))
      .attr("r", 5)
      .attr("fill", d=>d.attrs.filled? "black" : "white");
  
   /************************
   * modplus
   ******************/
  var g = svg.append("g")
    .attr("stroke", "black")
    .selectAll("g")
    .data(data.filter(d=>d.symb=="ModPlus"))
    .join("g")
    .attr("transform", (d,i) => `translate(${xScale(x(d))}, ${yScale(y(d))})`)
  g.append("circle")
      .attr("fill", "none")
      .attr("r", 7)
      .attr("cy", 0)
      .attr("cx", 0)
  g.append("path")
        .attr("d", "M 0 -7 l 0 14 M -7 0 l 14 0")
  
  /************************
   * text
   ******************/
  svg.append("g")
    .selectAll("text")
    .data(data.filter(d=>d.symb==="Text"))
    .join("text")
      .attr("text-anchor", "middle")
      .attr("x", d => xScale(x(d)))
      .attr("y", d => yScale(y(d)))
      .text(d => d.attrs.text);
  
  /************************
   * rarrowtextbox
   ******************/
  var g = svg.append("g")
    .selectAll("g")
    .data(data.filter(d=>d.symb=="RarrowTextBox"))
    .join("g")
    .attr("transform", (d,i) => `translate(${xScale(x(d))}, ${yScale(y(d))})`)
  g.append("polygon")
      .attr("fill", "white").attr("stroke", "black")
      .attr("points", "-25,-10 15,-10 25,0 15,10 -25,10")
      .attr("y", -10)
      .attr("x", -25)
  g.append("text")
        .attr("text-anchor", "middle")
        .attr("dominant-baseline", "middle")
        .text(d => d.attrs.text);
  
  /************************
   * rarrowtextbox
   ******************/
  var g = svg.append("g")
    .selectAll("g")
    .data(data.filter(d=>d.symb=="LarrowTextBox"))
    .join("g")
    .attr("transform", (d,i) => `translate(${xScale(x(d))}, ${yScale(y(d))})`)
  g.append("polygon")
      .attr("fill", "white").attr("stroke", "black")
      .attr("points", "-25,0 -15,-10 25,-10 25,10 -15,10")
      .attr("y", -10)
      .attr("x", -25)
  g.append("text")
        .attr("text-anchor", "middle")
        .attr("dominant-baseline", "middle")
        .text(d => d.attrs.text);

  // rects.attr("width", function() { return this.parentNode.childNodes[1].getComputedTextLength() + 10})
  // rects.attr("x", function(){return -(this.parentNode.childNodes[1].getComputedTextLength() + 10)/2})
  return svg.node();
}

d3.json("unary.json")
  .then(data => Scatterplot(data.soqs, data.vlines, {x: (d)=>d.x, y: (d)=>d.y}))
  .then(svg => d3.select("#content").append((d,i,nodes)=>svg));
