
import * as d3 from "d3";

interface CanvasProps {
    width: number; height: number;
    marginTop: number; marginRight: number;
    marginBottom: number; marginLeft: number;
}

export const canvas: CanvasProps = {
    width: 800,
    height: 600,
    marginTop: 20,
    marginRight: 30,
    marginBottom: 30,
    marginLeft: 30,
}

export function get_scales(canvas: CanvasProps) {
    // The x (horizontal position) scale.
    const x = d3.scaleLinear()
        .domain([0, 100])
        .range([canvas.marginLeft, canvas.width - canvas.marginRight]);
    const x_hat = x(1) - x(0);

    // The y (vertical position) scale.
    const y = d3.scaleLinear()
        .domain([0, 100])
        .range([canvas.marginTop, canvas.height - canvas.marginBottom]);
    const y_hat = y(1) - y(0);

    return { x: x, x_hat: x_hat, y: y, y_hat: y_hat }
}

export function get_svg(canvas: CanvasProps, x: d3.AxisScale<number>, y: d3.AxisScale<number>): d3.Selection<SVGSVGElement, undefined, null, undefined> {
    // Create the SVG container.
    const svg = d3.create("svg")
        .attr("width", canvas.width)
        .attr("height", canvas.height);

    // Add the x-axis.
    svg.append("g")
        .attr("transform", `translate(0,${canvas.height - canvas.marginBottom})`)
        .call(d3.axisBottom(x));

    // Add the y-axis.
    svg.append("g")
        .attr("transform", `translate(${canvas.marginLeft},0)`)
        .call(d3.axisLeft(y));

    return svg;
}