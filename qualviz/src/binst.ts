
export interface RegInstance {
  label: string;
  side: string;
}

export interface Bloq {
  pretty_name: string;
}

export interface BinstBox {
  i: number;
  bloq: Bloq;
  reg_instances: Array<RegInstance>;
  x: number;
  y: number;
  gen_i: number;
  fx?: number;
  fy?: number;
}

export interface Soquet {
  binst: BinstBox;
  soq_i: number;
}

export type Cxn = [Soquet, Soquet];