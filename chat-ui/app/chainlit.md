# Welcome to the Raj AI Lab

Type a prompt and the **LangGraph router** will:

1. **Classify** it (trivial / reasoning / hard) on the always-on Llama 8B.
2. **Ensure** the chosen target model is warm — scaling Karpenter-managed GPU nodes from zero if needed.
3. **Execute** on the routed model.

You'll see each step rendered live with timings, a JSON sidebar with the routing telemetry, and a deep-link to the matching trace in Langfuse.

Use `/upload` to upload a document (ingestion pipeline coming in the next iteration).
