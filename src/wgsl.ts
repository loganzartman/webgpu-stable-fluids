import dedent from "dedent";

export function wgsl(strings: TemplateStringsArray, ...values: unknown[]) {
  return dedent(strings, ...values);
}
