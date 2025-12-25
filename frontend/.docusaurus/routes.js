import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/__docusaurus/debug',
    component: ComponentCreator('/__docusaurus/debug', 'd4e'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/config',
    component: ComponentCreator('/__docusaurus/debug/config', 'f4f'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/content',
    component: ComponentCreator('/__docusaurus/debug/content', '0ac'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/globalData',
    component: ComponentCreator('/__docusaurus/debug/globalData', 'e0e'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/metadata',
    component: ComponentCreator('/__docusaurus/debug/metadata', '3a3'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/registry',
    component: ComponentCreator('/__docusaurus/debug/registry', 'fc2'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/routes',
    component: ComponentCreator('/__docusaurus/debug/routes', 'bf3'),
    exact: true
  },
  {
    path: '/docs',
    component: ComponentCreator('/docs', '69a'),
    routes: [
      {
        path: '/docs',
        component: ComponentCreator('/docs', 'a0f'),
        routes: [
          {
            path: '/docs',
            component: ComponentCreator('/docs', '938'),
            routes: [
              {
                path: '/docs/',
                component: ComponentCreator('/docs/', 'b00'),
                exact: true
              },
              {
                path: '/docs/',
                component: ComponentCreator('/docs/', 'da5'),
                exact: true
              },
              {
                path: '/docs/BOOK_COMPLETION_STATUS',
                component: ComponentCreator('/docs/BOOK_COMPLETION_STATUS', 'bff'),
                exact: true
              },
              {
                path: '/docs/book_summary',
                component: ComponentCreator('/docs/book_summary', '7f4'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/chapter1',
                component: ComponentCreator('/docs/chapter1', 'b63'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/chapter10',
                component: ComponentCreator('/docs/chapter10', 'f65'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/chapter2',
                component: ComponentCreator('/docs/chapter2', '2c9'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/chapter3',
                component: ComponentCreator('/docs/chapter3', '1a8'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/chapter4',
                component: ComponentCreator('/docs/chapter4', '342'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/chapter5',
                component: ComponentCreator('/docs/chapter5', '2d2'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/chapter6',
                component: ComponentCreator('/docs/chapter6', '6ff'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/chapter7',
                component: ComponentCreator('/docs/chapter7', '2f4'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/chapter8',
                component: ComponentCreator('/docs/chapter8', 'fb8'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/chapter9',
                component: ComponentCreator('/docs/chapter9', '584'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/intro/',
                component: ComponentCreator('/docs/intro/', 'c90'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/table_of_contents',
                component: ComponentCreator('/docs/table_of_contents', '60a'),
                exact: true,
                sidebar: "tutorialSidebar"
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '/',
    component: ComponentCreator('/', 'f7d'),
    exact: true
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
