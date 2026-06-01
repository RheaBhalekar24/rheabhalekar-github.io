const cards = document.querySelectorAll('.feature,.pipeline-step,.video-card,figure,.code-card');
const observer = new IntersectionObserver(entries => {
  entries.forEach(e => { if (e.isIntersecting) e.target.classList.add('show'); });
}, {threshold: .12});
cards.forEach(c => observer.observe(c));
