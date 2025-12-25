// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    {
      type: 'doc',
      id: 'book_summary',
      label: 'Book Summary',
    },
    {
      type: 'doc',
      id: 'table_of_contents',
      label: 'Table of Contents',
    },
    {
      type: 'category',
      label: 'Getting Started',
      items: [
        {
          type: 'doc',
          id: 'intro/intro',
        },
      ],
    },
    {
      type: 'category',
      label: 'Core Chapters',
      items: [
        {
          type: 'doc',
          id: 'chapter1',
          label: 'Chapter 1: Introduction to Physical AI and Embodied Intelligence',
        },
        {
          type: 'doc',
          id: 'chapter2',
          label: 'Chapter 2: ROS 2 Fundamentals for Humanoid Robots',
        },
        {
          type: 'doc',
          id: 'chapter3',
          label: 'Chapter 3: Simulation Environments - Gazebo, Isaac, Unity',
        },
        {
          type: 'doc',
          id: 'chapter4',
          label: 'Chapter 4: Perception Systems for Physical AI',
        },
        {
          type: 'doc',
          id: 'chapter5',
          label: 'Chapter 5: Planning and Navigation in Physical Space',
        },
        {
          type: 'doc',
          id: 'chapter6',
          label: 'Chapter 6: Control Systems and Actuation',
        },
        {
          type: 'doc',
          id: 'chapter7',
          label: 'Chapter 7: Vision-Language-Action Pipeline',
        },
        {
          type: 'doc',
          id: 'chapter8',
          label: 'Chapter 8: Humanoid-Specific Challenges and Solutions',
        },
        {
          type: 'doc',
          id: 'chapter9',
          label: 'Chapter 9: Simulation to Real-World Deployment',
        },
        {
          type: 'doc',
          id: 'chapter10',
          label: 'Chapter 10: Capstone - Complete Embodied AI System',
        },
      ],
    },
  ],
};

module.exports = sidebars;