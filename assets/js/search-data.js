// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "about",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-blog",
          title: "blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "post-context-engineering-for-agents-three-levels-of-disclosure",
        
          title: "Context Engineering for Agents: Three Levels of Disclosure",
        
        description: "",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/context-engineering-in-practice/";
          
        },
      },{id: "post-recursive-language-models-code-execution-60-accuracy-on-browsecomp-plus-no-embeddings",
        
          title: "Recursive Language Models + Code Execution: 60% accuracy on BrowseComp Plus (no embeddings)...",
        
        description: "",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/recursive-lm-code-execution/";
          
        },
      },{id: "post-recursive-language-models-reduce-context-rot-and-2-5-accuracy-on-browsecomp-plus-at-2-6-latency",
        
          title: "Recursive Language Models reduce context rot and 2.5× accuracy on BrowseComp‑Plus (at 2.6×...",
        
        description: "",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/recursive-language-models/";
          
        },
      },{id: "post-exploring-continuous-learning-reasoning-bank-recursive-language-models",
        
          title: "Exploring Continuous Learning: Reasoning Bank + Recursive Language Models",
        
        description: "",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/anti-patterns-as-guardrails/";
          
        },
      },{
        id: 'social-rss',
        title: 'RSS Feed',
        section: 'Socials',
        handler: () => {
          window.open("/feed.xml", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
