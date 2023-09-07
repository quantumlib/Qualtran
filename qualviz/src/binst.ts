
export interface Soquet {
  label:string;
  side:string;
}

export interface Bloq{
  pretty_name: string;
}

export interface BinstBox {
  i: number;
  bloq: Bloq;
  soqs: Array<Soquet>;
  x: number;
  y: number;
}

export interface Port {
  binst_i: number;
  soq_i: number;
}

export type Cxn = [Port, Port];