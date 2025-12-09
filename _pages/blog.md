---
layout: page
permalink: /
title: blog
nav: false
pagination:
  enabled: true
  collection: posts
  permalink: /page/:num/
  per_page: 5
  sort_field: date
  sort_reverse: true
  trail:
    before: 1 # The number of links before the current page
    after: 3 # The number of links after the current page
---

{% if page.pagination.enabled %}
{% assign postlist = paginator.posts %}
{% else %}
{% assign postlist = site.posts %}
{% endif %}

<ul class="post-list">
  {% for post in postlist %}
    <li class="mb-4">
      <h2 class="h4 mb-1">
        <a class="post-title" href="{{ post.url | relative_url }}">{{ post.title }}</a>
      </h2>
      <p class="post-meta mb-0">{{ post.date | date: '%B %d, %Y' }}</p>
    </li>
  {% endfor %}
</ul>

{% if page.pagination.enabled %}
{% include pagination.liquid %}
{% endif %}
