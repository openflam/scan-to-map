import type * as React from "react";

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

export {};
