// main.jsx — React Entry Point
// What: Mounts the React app into the HTML page.
// Why:  React needs a root DOM element to render into.

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./App.css";
import App from "./App.jsx";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <App />
  </StrictMode>
);
