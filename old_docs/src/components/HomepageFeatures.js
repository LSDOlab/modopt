import React from 'react';
import clsx from 'clsx';
import styles from './HomepageFeatures.module.css';

const FeatureList = [
  {
    title: 'Educational',
    Svg: require('../../static/img/lightbulb_icon.svg').default,
    description: (
      <>Breaks down optimization algorithms into self-contained components such as line-searches and merit functions.
        Students can develop new or modified versions of these components and compare its performance with a standard algorithm.
        {/* Students can work on developing new or modified verisons of some of these independent components as part of assignments and */}
        {/* compare their results with the original algorithm. */}
      </>
    ),
  },

  {
    title: 'Accelerated Optimizer Development',
    Svg: require('../../static/img/rapid_icon.svg').default,
    description: (
      <>
        {/* Speed-up optimizer development by breaking down optimization algorithms into independent components and */}
        Speed-up optimizer development by reusing standard components already available
        and testing with problems from test-suites.
      </>
    ),
  },
  {
    title: 'Scalability',
    Svg: require('../../static/img/scalability_icon.svg').default,
    description: (
      <>Sparsity structures could be specified for sub-Jacobians and
        the framework automatically computes the full Jacobian in the specified format in the most efficient way.</>
    ),
  },
];

function Feature({ Svg, title, description }) {
  return (
    <div className={clsx('col col--4')} >
      <div className="text--center" >
        <Svg className={styles.featureSvg} alt={title} />
      </div>
      <div className="text--center padding-horiz--md">
        <h3> {title} </h3>
        <p> {description} </p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (<
        section className={styles.features} >
    <
        div className="container" >
      <
        div className="row" > {
          FeatureList.map((props, idx) => (<
            Feature key={idx} {...props}
          />
          ))
        } <
        /div> < /
        div > <
        /section>
        );
}