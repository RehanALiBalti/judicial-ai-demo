export function SkeletonLine({ width = "100%" }) {
  return <div className="skeleton skeleton-line" style={{ width }} />;
}

export function SkeletonCard() {
  return (
    <div className="skeleton-card">
      <SkeletonLine width="70%" />
      <SkeletonLine width="45%" />
      <div className="skeleton-grid">
        <SkeletonLine />
        <SkeletonLine />
        <SkeletonLine />
      </div>
    </div>
  );
}

export function SkeletonStats() {
  return (
    <div className="skeleton-stats">
      {[1, 2, 3].map((i) => (
        <div key={i} className="skeleton-stat" />
      ))}
    </div>
  );
}
