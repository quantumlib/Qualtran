
export interface Soquet {
  thru?: string;
  left?: string;
  right?: string;
}

export interface BinstBox {
  x: number;
  y: number;
  title: string;
  soqs: Array<Soquet>;
}

export interface Port {
  binst_i: number;
  soq_i: number;
}

export type Cxn = [Port, Port];