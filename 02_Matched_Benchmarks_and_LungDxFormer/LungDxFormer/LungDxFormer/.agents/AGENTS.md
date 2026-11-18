# Professional Coding Standards and Automatic Skill Invocation

- **Universal Professionalism**: Every time the user requests code (e.g., using the term "vibecode" or any other request for code), the assistant must automatically apply the suite of installed agent‑skills—whether they are from `agent‑skills`, `skills`, or any future plugins—without requiring explicit slash commands.
- **Workflow Enforcement**: The assistant will implicitly follow a full development lifecycle:
  1. **Planning** – break the task into a clear implementation plan.
  2. **Specification** – write a concise spec before any implementation.
  3. **Implementation** – produce clean, well‑documented code.
  4. **Testing** – include unit‑tests or integration‑tests as appropriate.
  5. **Review** – apply a five‑axis code review (using `code-review-and-quality`).
  6. **Simplification** – run `code‑simplify` if needed.
  7. **Ship** – ensure the code is ready for production.
- **Future‑Proof** – As new plugins are installed, they will be automatically considered in this pipeline.
- **No Slashes Required** – The assistant must not wait for a slash command; the professional pipeline is always applied.
