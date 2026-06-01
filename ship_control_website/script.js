const observer = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) entry.target.classList.add('visible');
  });
}, { threshold: 0.12 });

document.querySelectorAll('.section, figure, .pipeline article, .loss-grid article').forEach((el) => {
  el.classList.add('reveal');
  observer.observe(el);
});
