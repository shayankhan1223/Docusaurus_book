import React from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';

const FeatureList = [
  {
    title: 'AI-Powered Search',
    description: (
      <>
        Intelligent search powered by AI that understands context and provides
        relevant documentation answers.
      </>
    ),
  },
  {
    title: 'Easy Navigation',
    description: (
      <>
        Intuitive navigation with multilingual support to help you find what you need.
      </>
    ),
  },
  {
    title: 'Smart Assistance',
    description: (
      <>
        Get help with documentation queries through our AI assistant.
      </>
    ),
  },
];

function Feature({title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className="features">
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}