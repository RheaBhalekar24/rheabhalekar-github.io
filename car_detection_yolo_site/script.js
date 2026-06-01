const revealItems = document.querySelectorAll('.section, .stats article, .pipeline-grid article, .gallery figure, .code-grid div');
revealItems.forEach(item => item.classList.add('reveal'));

const observer = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.classList.add('visible');
      observer.unobserve(entry.target);
    }
  });
}, { threshold: 0.12 });

revealItems.forEach(item => observer.observe(item));
