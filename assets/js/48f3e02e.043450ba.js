"use strict";(self.webpackChunkmy_website=self.webpackChunkmy_website||[]).push([[325],{3905:(e,t,n)=>{n.d(t,{Zo:()=>s,kt:()=>u});var r=n(7294);function o(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function i(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function a(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?i(Object(n),!0).forEach((function(t){o(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):i(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function l(e,t){if(null==e)return{};var n,r,o=function(e,t){if(null==e)return{};var n,r,o={},i=Object.keys(e);for(r=0;r<i.length;r++)n=i[r],t.indexOf(n)>=0||(o[n]=e[n]);return o}(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(r=0;r<i.length;r++)n=i[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(o[n]=e[n])}return o}var p=r.createContext({}),m=function(e){var t=r.useContext(p),n=t;return e&&(n="function"==typeof e?e(t):a(a({},t),e)),n},s=function(e){var t=m(e.components);return r.createElement(p.Provider,{value:t},e.children)},c={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},d=r.forwardRef((function(e,t){var n=e.components,o=e.mdxType,i=e.originalType,p=e.parentName,s=l(e,["components","mdxType","originalType","parentName"]),d=m(n),u=o,b=d["".concat(p,".").concat(u)]||d[u]||c[u]||i;return n?r.createElement(b,a(a({ref:t},s),{},{components:n})):r.createElement(b,a({ref:t},s))}));function u(e,t){var n=arguments,o=t&&t.mdxType;if("string"==typeof e||o){var i=n.length,a=new Array(i);a[0]=d;var l={};for(var p in t)hasOwnProperty.call(t,p)&&(l[p]=t[p]);l.originalType=e,l.mdxType="string"==typeof e?e:o,a[1]=l;for(var m=2;m<i;m++)a[m]=n[m];return r.createElement.apply(null,a)}return r.createElement.apply(null,n)}d.displayName="MDXCreateElement"},7187:(e,t,n)=>{n.r(t),n.d(t,{contentTitle:()=>a,default:()=>s,frontMatter:()=>i,metadata:()=>l,toc:()=>p});var r=n(7462),o=(n(7294),n(3905));const i={sidebar_position:1},a=void 0,l={unversionedId:"optimizers_available/optimizers_available",id:"optimizers_available/optimizers_available",isDocsHomePage:!1,title:"optimizers_available",description:"Standard optimization algorithms implemented",source:"@site/docs/optimizers_available/optimizers_available.mdx",sourceDirName:"optimizers_available",slug:"/optimizers_available/optimizers_available",permalink:"/modopt/docs/optimizers_available/optimizers_available",editUrl:"https://github.com/lsdolab/modopt/edit/main/website/docs/optimizers_available/optimizers_available.mdx",tags:[],version:"current",sidebarPosition:1,frontMatter:{sidebar_position:1},sidebar:"tutorialSidebar",previous:{title:"Benchmarking",permalink:"/modopt/docs/benchmarking"},next:{title:"SLSQP",permalink:"/modopt/docs/optimizers_available/SLSQP"}},p=[{value:"Standard optimization algorithms implemented",id:"standard-optimization-algorithms-implemented",children:[{value:"1. Steepest-Descent",id:"1-steepest-descent",children:[]},{value:"2. Newton",id:"2-newton",children:[]},{value:"3. Quasi-Newton",id:"3-quasi-newton",children:[]},{value:"4. Newton-Lagrange",id:"4-newton-lagrange",children:[]},{value:"5. l2-Penalty",id:"5-l2-penalty",children:[]}]},{value:"Usage instructions",id:"usage-instructions",children:[]}],m={toc:p};function s(e){let{components:t,...n}=e;return(0,o.kt)("wrapper",(0,r.Z)({},m,n,{components:t,mdxType:"MDXLayout"}),(0,o.kt)("h2",{id:"standard-optimization-algorithms-implemented"},"Standard optimization algorithms implemented"),(0,o.kt)("p",null,"Currently, modOpt has a fully transparent library of optimization algorithms\nimplemented for different types of optimization problems.\nThe following are the algorithms implemented:"),(0,o.kt)("h3",{id:"1-steepest-descent"},"1. Steepest-Descent"),(0,o.kt)("p",null,"The implementation can be found\n",(0,o.kt)("strong",{parentName:"p"},(0,o.kt)("a",{parentName:"strong",href:"https://github.com/LSDOlab/modopt/blob/main/modopt/core/optimization_algorithms/steepest_descent.py"},"here")),"."),(0,o.kt)("h3",{id:"2-newton"},"2. Newton"),(0,o.kt)("p",null,"The implementation can be found\n",(0,o.kt)("strong",{parentName:"p"},(0,o.kt)("a",{parentName:"strong",href:"https://github.com/LSDOlab/modopt/blob/main/modopt/core/optimization_algorithms/newton.py"},"here")),"."),(0,o.kt)("h3",{id:"3-quasi-newton"},"3. Quasi-Newton"),(0,o.kt)("p",null,"The implementation can be found\n",(0,o.kt)("strong",{parentName:"p"},(0,o.kt)("a",{parentName:"strong",href:"https://github.com/LSDOlab/modopt/blob/main/modopt/core/optimization_algorithms/quasi_newton.py"},"here")),"."),(0,o.kt)("h3",{id:"4-newton-lagrange"},"4. Newton-Lagrange"),(0,o.kt)("p",null,"The implementation can be found\n",(0,o.kt)("strong",{parentName:"p"},(0,o.kt)("a",{parentName:"strong",href:"https://github.com/LSDOlab/modopt/blob/main/modopt/core/optimization_algorithms/newton_lagrange.py"},"here")),"."),(0,o.kt)("h3",{id:"5-l2-penalty"},"5. l2-Penalty"),(0,o.kt)("p",null,"The implementation can be found\n",(0,o.kt)("strong",{parentName:"p"},(0,o.kt)("a",{parentName:"strong",href:"https://github.com/LSDOlab/modopt/blob/main/modopt/core/optimization_algorithms/quadratic_penalty_eq.py"},"here")),"."),(0,o.kt)("h2",{id:"usage-instructions"},"Usage instructions"),(0,o.kt)("p",null,"In order to use these algorithms with any of the  problems written using\nthe ",(0,o.kt)("strong",{parentName:"p"},"Problem()")," class, you should first import your problem from the corresponding file\nand also import the optimizer of your choice from the library.\nAfter that, set tolerances and other parameters for the chosen optimizer.\nSolve the problem and then print results."),(0,o.kt)("p",null,"An example is shown below for the SteepestDescent optimizer."),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-py"},"\nfrom my_problem import MyProblem\n\nfrom modopt.optimization_algorithms import SteepestDescent\n\ntol = 1E-8\nmax_itr = 500\n\nprob = MyProblem()\n\noptimizer = SteepestDescent(\n    prob,\n    opt_tol=tol,\n    max_itr=max_itr,\n    )\n\noptimizer.check_first_derivatives(prob.x.get_data())\noptimizer.solve()\noptimizer.print_results(summary_table=True)\n")))}s.isMDXComponent=!0}}]);