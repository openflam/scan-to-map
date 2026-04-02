import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import AppBenchmark from "./AppBenchmark";

import "bootstrap/dist/css/bootstrap.min.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <AppBenchmark />
  </StrictMode>,
);
