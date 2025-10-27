import type * as React from "react";

// Define BoundingBox interface
export interface BoundingBox {
  x_min: number;
  y_min: number;
  z_min: number;
  x_max: number;
  y_max: number;
  z_max: number;
}

// Define SearchResult interface
export interface SearchResult {
  boundingBox: BoundingBox;
  reason: string;
}

// Define custom attributes for babylon-viewer element
interface HTML3DElementAttributes
  extends React.DetailedHTMLProps<
    React.HTMLAttributes<HTMLElement>,
    HTMLElement
  > {
  source?: string;
  environment?: string;
}

declare global {
  namespace JSX {
    interface IntrinsicElements {
      "babylon-viewer": HTML3DElementAttributes;
    }
  }
}

export { };
