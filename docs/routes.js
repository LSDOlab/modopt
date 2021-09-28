
import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/mod_opt/',
    component: ComponentCreator('/mod_opt/', 'ef8'),
    exact: true
  },
  {
    path: '/mod_opt/__docusaurus/debug',
    component: ComponentCreator('/mod_opt/__docusaurus/debug', '731'),
    exact: true
  },
  {
    path: '/mod_opt/__docusaurus/debug/config',
    component: ComponentCreator('/mod_opt/__docusaurus/debug/config', '6d7'),
    exact: true
  },
  {
    path: '/mod_opt/__docusaurus/debug/content',
    component: ComponentCreator('/mod_opt/__docusaurus/debug/content', 'e90'),
    exact: true
  },
  {
    path: '/mod_opt/__docusaurus/debug/globalData',
    component: ComponentCreator('/mod_opt/__docusaurus/debug/globalData', '3d6'),
    exact: true
  },
  {
    path: '/mod_opt/__docusaurus/debug/metadata',
    component: ComponentCreator('/mod_opt/__docusaurus/debug/metadata', '976'),
    exact: true
  },
  {
    path: '/mod_opt/__docusaurus/debug/registry',
    component: ComponentCreator('/mod_opt/__docusaurus/debug/registry', 'fcc'),
    exact: true
  },
  {
    path: '/mod_opt/__docusaurus/debug/routes',
    component: ComponentCreator('/mod_opt/__docusaurus/debug/routes', '7b4'),
    exact: true
  },
  {
    path: '/mod_opt/blog/archive',
    component: ComponentCreator('/mod_opt/blog/archive', '8cb'),
    exact: true
  },
  {
    path: '/mod_opt/community',
    component: ComponentCreator('/mod_opt/community', '36e'),
    exact: true
  },
  {
    path: '/mod_opt/docs/tags',
    component: ComponentCreator('/mod_opt/docs/tags', '8fa'),
    exact: true
  },
  {
    path: '/mod_opt/faq',
    component: ComponentCreator('/mod_opt/faq', 'ee8'),
    exact: true
  },
  {
    path: '/mod_opt/news',
    component: ComponentCreator('/mod_opt/news', '9e0'),
    exact: true
  },
  {
    path: '/mod_opt/publications',
    component: ComponentCreator('/mod_opt/publications', '63f'),
    exact: true
  },
  {
    path: '/mod_opt/docs',
    component: ComponentCreator('/mod_opt/docs', 'cd2'),
    routes: [
      {
        path: '/mod_opt/docs/benchmarking',
        component: ComponentCreator('/mod_opt/docs/benchmarking', '700'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/mod_opt/docs/getting_started/coupling_optimizer_with_problem',
        component: ComponentCreator('/mod_opt/docs/getting_started/coupling_optimizer_with_problem', '1d4'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/mod_opt/docs/getting_started/new_optimizer',
        component: ComponentCreator('/mod_opt/docs/getting_started/new_optimizer', '88c'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/mod_opt/docs/getting_started/new_problem',
        component: ComponentCreator('/mod_opt/docs/getting_started/new_problem', 'c97'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/mod_opt/docs/getting_started/simple_example',
        component: ComponentCreator('/mod_opt/docs/getting_started/simple_example', 'c6a'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/mod_opt/docs/installation_instructions',
        component: ComponentCreator('/mod_opt/docs/installation_instructions', '3ba'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/mod_opt/docs/interfacing_existing_optimizers',
        component: ComponentCreator('/mod_opt/docs/interfacing_existing_optimizers', '6c1'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/mod_opt/docs/introduction',
        component: ComponentCreator('/mod_opt/docs/introduction', '995'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/mod_opt/docs/more_examples',
        component: ComponentCreator('/mod_opt/docs/more_examples', 'e51'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/mod_opt/docs/standard_optimization_algorithms',
        component: ComponentCreator('/mod_opt/docs/standard_optimization_algorithms', '715'),
        exact: true,
        'sidebar': "tutorialSidebar"
      }
    ]
  },
  {
    path: '*',
    component: ComponentCreator('*')
  }
];
