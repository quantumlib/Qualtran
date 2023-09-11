
export interface RegInstance {
  i: number;
  label: string;
  side: string;
  x: number;
  y: number;
  binst: Binst;
  fx?: number;
  fy?: number;
}

export interface Bloq {
  pretty_name: string;
}

export interface Binst {
  i: number;
  bloq: Bloq;
  gen_i: number;
  reg_instances: Array<RegInstance>;
  x0?: number;
  x1?: number;
  y0?: number;
  y1?: number;
}


export type Cxn = [RegInstance, RegInstance];