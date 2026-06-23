export default function Spinner({ size = "md", className = "" }) {
  return <span className={`spinner spinner-${size} ${className}`.trim()} aria-hidden="true" />;
}
