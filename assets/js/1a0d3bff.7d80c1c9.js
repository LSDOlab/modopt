"use strict";(self.webpackChunkmy_website=self.webpackChunkmy_website||[]).push([[630],{5680:(a,e,s)=>{s.d(e,{xA:()=>g,yg:()=>c});var t=s(6540);function n(a,e,s){return e in a?Object.defineProperty(a,e,{value:s,enumerable:!0,configurable:!0,writable:!0}):a[e]=s,a}function m(a,e){var s=Object.keys(a);if(Object.getOwnPropertySymbols){var t=Object.getOwnPropertySymbols(a);e&&(t=t.filter((function(e){return Object.getOwnPropertyDescriptor(a,e).enumerable}))),s.push.apply(s,t)}return s}function p(a){for(var e=1;e<arguments.length;e++){var s=null!=arguments[e]?arguments[e]:{};e%2?m(Object(s),!0).forEach((function(e){n(a,e,s[e])})):Object.getOwnPropertyDescriptors?Object.defineProperties(a,Object.getOwnPropertyDescriptors(s)):m(Object(s)).forEach((function(e){Object.defineProperty(a,e,Object.getOwnPropertyDescriptor(s,e))}))}return a}function r(a,e){if(null==a)return{};var s,t,n=function(a,e){if(null==a)return{};var s,t,n={},m=Object.keys(a);for(t=0;t<m.length;t++)s=m[t],e.indexOf(s)>=0||(n[s]=a[s]);return n}(a,e);if(Object.getOwnPropertySymbols){var m=Object.getOwnPropertySymbols(a);for(t=0;t<m.length;t++)s=m[t],e.indexOf(s)>=0||Object.prototype.propertyIsEnumerable.call(a,s)&&(n[s]=a[s])}return n}var i=t.createContext({}),l=function(a){var e=t.useContext(i),s=e;return a&&(s="function"==typeof a?a(e):p(p({},e),a)),s},g=function(a){var e=l(a.components);return t.createElement(i.Provider,{value:e},a.children)},o="mdxType",N={inlineCode:"code",wrapper:function(a){var e=a.children;return t.createElement(t.Fragment,{},e)}},y=t.forwardRef((function(a,e){var s=a.components,n=a.mdxType,m=a.originalType,i=a.parentName,g=r(a,["components","mdxType","originalType","parentName"]),o=l(s),y=n,c=o["".concat(i,".").concat(y)]||o[y]||N[y]||m;return s?t.createElement(c,p(p({ref:e},g),{},{components:s})):t.createElement(c,p({ref:e},g))}));function c(a,e){var s=arguments,n=e&&e.mdxType;if("string"==typeof a||n){var m=s.length,p=new Array(m);p[0]=y;var r={};for(var i in e)hasOwnProperty.call(e,i)&&(r[i]=e[i]);r.originalType=a,r[o]="string"==typeof a?a:n,p[1]=r;for(var l=2;l<m;l++)p[l]=s[l];return t.createElement.apply(null,p)}return t.createElement.apply(null,s)}y.displayName="MDXCreateElement"},5778:(a,e,s)=>{s.r(e),s.d(e,{contentTitle:()=>p,default:()=>o,frontMatter:()=>m,metadata:()=>r,toc:()=>i});var t=s(8168),n=(s(6540),s(5680));const m={title:"Building Custom Optimizers",sidebar_position:9},p="Building Custom Optimizers",r={unversionedId:"building_custom_optimizer",id:"building_custom_optimizer",isDocsHomePage:!1,title:"Building Custom Optimizers",description:"Here we look at the steepest descent algorithm for unconstrained problems.",source:"@site/docs/building_custom_optimizer.mdx",sourceDirName:".",slug:"/building_custom_optimizer",permalink:"/modopt/docs/building_custom_optimizer",editUrl:"https://github.com/lsdolab/modopt/edit/main/website/docs/building_custom_optimizer.mdx",tags:[],version:"current",sidebarPosition:9,frontMatter:{title:"Building Custom Optimizers",sidebar_position:9},sidebar:"tutorialSidebar",previous:{title:"SNOPT",permalink:"/modopt/docs/optimizers_available/SNOPT"}},i=[],l={toc:i},g="wrapper";function o(a){let{components:e,...s}=a;return(0,n.yg)(g,(0,t.A)({},l,s,{components:e,mdxType:"MDXLayout"}),(0,n.yg)("h1",{id:"building-custom-optimizers"},"Building Custom Optimizers"),(0,n.yg)("p",null,"Here we look at the ",(0,n.yg)("strong",{parentName:"p"},"steepest descent")," algorithm for unconstrained problems. "),(0,n.yg)("p",null,"For a general unconstrained optimization problem stated as: "),(0,n.yg)("div",{className:"math math-display"},(0,n.yg)("span",{parentName:"div",className:"katex-display"},(0,n.yg)("span",{parentName:"span",className:"katex"},(0,n.yg)("span",{parentName:"span",className:"katex-mathml"},(0,n.yg)("math",{parentName:"span",xmlns:"http://www.w3.org/1998/Math/MathML",display:"block"},(0,n.yg)("semantics",{parentName:"math"},(0,n.yg)("mrow",{parentName:"semantics"},(0,n.yg)("mi",{parentName:"mrow"},(0,n.yg)("munder",{parentName:"mi"},(0,n.yg)("mo",{parentName:"munder"},(0,n.yg)("mtext",{parentName:"mo"},"minimize")),(0,n.yg)("mrow",{parentName:"munder"},(0,n.yg)("mi",{parentName:"mrow"},"x"),(0,n.yg)("mo",{parentName:"mrow"},"\u2208"),(0,n.yg)("msup",{parentName:"mrow"},(0,n.yg)("mi",{parentName:"msup",mathvariant:"double-struck"},"R"),(0,n.yg)("mi",{parentName:"msup",mathvariant:"double-struck"},"n"))))),(0,n.yg)("mspace",{parentName:"mrow",width:"1em"}),(0,n.yg)("mi",{parentName:"mrow"},"f"),(0,n.yg)("mo",{parentName:"mrow",stretchy:"false"},"("),(0,n.yg)("mi",{parentName:"mrow"},"x"),(0,n.yg)("mo",{parentName:"mrow",stretchy:"false"},")")),(0,n.yg)("annotation",{parentName:"semantics",encoding:"application/x-tex"},"\\underset{x \\in \\mathbb{R^n}}{\\text{minimize}} \\quad f(x)")))),(0,n.yg)("span",{parentName:"span",className:"katex-html","aria-hidden":"true"},(0,n.yg)("span",{parentName:"span",className:"base"},(0,n.yg)("span",{parentName:"span",className:"strut",style:{height:"1.525593em",verticalAlign:"-0.775593em"}}),(0,n.yg)("span",{parentName:"span",className:"mord"},(0,n.yg)("span",{parentName:"span",className:"mop op-limits"},(0,n.yg)("span",{parentName:"span",className:"vlist-t vlist-t2"},(0,n.yg)("span",{parentName:"span",className:"vlist-r"},(0,n.yg)("span",{parentName:"span",className:"vlist",style:{height:"0.66786em"}},(0,n.yg)("span",{parentName:"span",style:{top:"-2.351777em",marginLeft:"0em"}},(0,n.yg)("span",{parentName:"span",className:"pstrut",style:{height:"3em"}}),(0,n.yg)("span",{parentName:"span",className:"sizing reset-size6 size3 mtight"},(0,n.yg)("span",{parentName:"span",className:"mord mtight"},(0,n.yg)("span",{parentName:"span",className:"mord mathnormal mtight"},"x"),(0,n.yg)("span",{parentName:"span",className:"mrel mtight"},"\u2208"),(0,n.yg)("span",{parentName:"span",className:"mord mtight"},(0,n.yg)("span",{parentName:"span",className:"mord mtight"},(0,n.yg)("span",{parentName:"span",className:"mord mathbb mtight"},"R"),(0,n.yg)("span",{parentName:"span",className:"msupsub"},(0,n.yg)("span",{parentName:"span",className:"vlist-t"},(0,n.yg)("span",{parentName:"span",className:"vlist-r"},(0,n.yg)("span",{parentName:"span",className:"vlist",style:{height:"0.5935428571428571em"}},(0,n.yg)("span",{parentName:"span",style:{top:"-2.786em",marginRight:"0.07142857142857144em"}},(0,n.yg)("span",{parentName:"span",className:"pstrut",style:{height:"2.5em"}}),(0,n.yg)("span",{parentName:"span",className:"sizing reset-size3 size1 mtight"},(0,n.yg)("span",{parentName:"span",className:"mord mathnormal mtight"},"n")))))))))))),(0,n.yg)("span",{parentName:"span",style:{top:"-3em"}},(0,n.yg)("span",{parentName:"span",className:"pstrut",style:{height:"3em"}}),(0,n.yg)("span",{parentName:"span"},(0,n.yg)("span",{parentName:"span",className:"mop"},(0,n.yg)("span",{parentName:"span",className:"mord text"},(0,n.yg)("span",{parentName:"span",className:"mord"},"minimize")))))),(0,n.yg)("span",{parentName:"span",className:"vlist-s"},"\u200b")),(0,n.yg)("span",{parentName:"span",className:"vlist-r"},(0,n.yg)("span",{parentName:"span",className:"vlist",style:{height:"0.775593em"}},(0,n.yg)("span",{parentName:"span"})))))),(0,n.yg)("span",{parentName:"span",className:"mspace",style:{marginRight:"1em"}}),(0,n.yg)("span",{parentName:"span",className:"mord mathnormal",style:{marginRight:"0.10764em"}},"f"),(0,n.yg)("span",{parentName:"span",className:"mopen"},"("),(0,n.yg)("span",{parentName:"span",className:"mord mathnormal"},"x"),(0,n.yg)("span",{parentName:"span",className:"mclose"},")")))))),(0,n.yg)("p",null,"the steepest descent algorithms computes the new iterate recursively by using the formula"),(0,n.yg)("div",{className:"math math-display"},(0,n.yg)("span",{parentName:"div",className:"katex-display"},(0,n.yg)("span",{parentName:"span",className:"katex"},(0,n.yg)("span",{parentName:"span",className:"katex-mathml"},(0,n.yg)("math",{parentName:"span",xmlns:"http://www.w3.org/1998/Math/MathML",display:"block"},(0,n.yg)("semantics",{parentName:"math"},(0,n.yg)("mrow",{parentName:"semantics"},(0,n.yg)("msub",{parentName:"mrow"},(0,n.yg)("mi",{parentName:"msub"},"x"),(0,n.yg)("mrow",{parentName:"msub"},(0,n.yg)("mi",{parentName:"mrow"},"k"),(0,n.yg)("mo",{parentName:"mrow"},"+"),(0,n.yg)("mn",{parentName:"mrow"},"1"))),(0,n.yg)("mo",{parentName:"mrow"},"="),(0,n.yg)("msub",{parentName:"mrow"},(0,n.yg)("mi",{parentName:"msub"},"x"),(0,n.yg)("mi",{parentName:"msub"},"k")),(0,n.yg)("mo",{parentName:"mrow"},"\u2212"),(0,n.yg)("mi",{parentName:"mrow",mathvariant:"normal"},"\u2207"),(0,n.yg)("mi",{parentName:"mrow"},"f"),(0,n.yg)("mo",{parentName:"mrow",stretchy:"false"},"("),(0,n.yg)("msub",{parentName:"mrow"},(0,n.yg)("mi",{parentName:"msub"},"x"),(0,n.yg)("mi",{parentName:"msub"},"k")),(0,n.yg)("mo",{parentName:"mrow",stretchy:"false"},")"),(0,n.yg)("mi",{parentName:"mrow",mathvariant:"normal"},".")),(0,n.yg)("annotation",{parentName:"semantics",encoding:"application/x-tex"},"x_{k+1} = x_{k} - \\nabla f(x_k) .")))),(0,n.yg)("span",{parentName:"span",className:"katex-html","aria-hidden":"true"},(0,n.yg)("span",{parentName:"span",className:"base"},(0,n.yg)("span",{parentName:"span",className:"strut",style:{height:"0.638891em",verticalAlign:"-0.208331em"}}),(0,n.yg)("span",{parentName:"span",className:"mord"},(0,n.yg)("span",{parentName:"span",className:"mord mathnormal"},"x"),(0,n.yg)("span",{parentName:"span",className:"msupsub"},(0,n.yg)("span",{parentName:"span",className:"vlist-t vlist-t2"},(0,n.yg)("span",{parentName:"span",className:"vlist-r"},(0,n.yg)("span",{parentName:"span",className:"vlist",style:{height:"0.3361079999999999em"}},(0,n.yg)("span",{parentName:"span",style:{top:"-2.5500000000000003em",marginLeft:"0em",marginRight:"0.05em"}},(0,n.yg)("span",{parentName:"span",className:"pstrut",style:{height:"2.7em"}}),(0,n.yg)("span",{parentName:"span",className:"sizing reset-size6 size3 mtight"},(0,n.yg)("span",{parentName:"span",className:"mord mtight"},(0,n.yg)("span",{parentName:"span",className:"mord mathnormal mtight",style:{marginRight:"0.03148em"}},"k"),(0,n.yg)("span",{parentName:"span",className:"mbin mtight"},"+"),(0,n.yg)("span",{parentName:"span",className:"mord mtight"},"1"))))),(0,n.yg)("span",{parentName:"span",className:"vlist-s"},"\u200b")),(0,n.yg)("span",{parentName:"span",className:"vlist-r"},(0,n.yg)("span",{parentName:"span",className:"vlist",style:{height:"0.208331em"}},(0,n.yg)("span",{parentName:"span"})))))),(0,n.yg)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.2777777777777778em"}}),(0,n.yg)("span",{parentName:"span",className:"mrel"},"="),(0,n.yg)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.2777777777777778em"}})),(0,n.yg)("span",{parentName:"span",className:"base"},(0,n.yg)("span",{parentName:"span",className:"strut",style:{height:"0.73333em",verticalAlign:"-0.15em"}}),(0,n.yg)("span",{parentName:"span",className:"mord"},(0,n.yg)("span",{parentName:"span",className:"mord mathnormal"},"x"),(0,n.yg)("span",{parentName:"span",className:"msupsub"},(0,n.yg)("span",{parentName:"span",className:"vlist-t vlist-t2"},(0,n.yg)("span",{parentName:"span",className:"vlist-r"},(0,n.yg)("span",{parentName:"span",className:"vlist",style:{height:"0.33610799999999996em"}},(0,n.yg)("span",{parentName:"span",style:{top:"-2.5500000000000003em",marginLeft:"0em",marginRight:"0.05em"}},(0,n.yg)("span",{parentName:"span",className:"pstrut",style:{height:"2.7em"}}),(0,n.yg)("span",{parentName:"span",className:"sizing reset-size6 size3 mtight"},(0,n.yg)("span",{parentName:"span",className:"mord mtight"},(0,n.yg)("span",{parentName:"span",className:"mord mathnormal mtight",style:{marginRight:"0.03148em"}},"k"))))),(0,n.yg)("span",{parentName:"span",className:"vlist-s"},"\u200b")),(0,n.yg)("span",{parentName:"span",className:"vlist-r"},(0,n.yg)("span",{parentName:"span",className:"vlist",style:{height:"0.15em"}},(0,n.yg)("span",{parentName:"span"})))))),(0,n.yg)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.2222222222222222em"}}),(0,n.yg)("span",{parentName:"span",className:"mbin"},"\u2212"),(0,n.yg)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.2222222222222222em"}})),(0,n.yg)("span",{parentName:"span",className:"base"},(0,n.yg)("span",{parentName:"span",className:"strut",style:{height:"1em",verticalAlign:"-0.25em"}}),(0,n.yg)("span",{parentName:"span",className:"mord"},"\u2207"),(0,n.yg)("span",{parentName:"span",className:"mord mathnormal",style:{marginRight:"0.10764em"}},"f"),(0,n.yg)("span",{parentName:"span",className:"mopen"},"("),(0,n.yg)("span",{parentName:"span",className:"mord"},(0,n.yg)("span",{parentName:"span",className:"mord mathnormal"},"x"),(0,n.yg)("span",{parentName:"span",className:"msupsub"},(0,n.yg)("span",{parentName:"span",className:"vlist-t vlist-t2"},(0,n.yg)("span",{parentName:"span",className:"vlist-r"},(0,n.yg)("span",{parentName:"span",className:"vlist",style:{height:"0.33610799999999996em"}},(0,n.yg)("span",{parentName:"span",style:{top:"-2.5500000000000003em",marginLeft:"0em",marginRight:"0.05em"}},(0,n.yg)("span",{parentName:"span",className:"pstrut",style:{height:"2.7em"}}),(0,n.yg)("span",{parentName:"span",className:"sizing reset-size6 size3 mtight"},(0,n.yg)("span",{parentName:"span",className:"mord mathnormal mtight",style:{marginRight:"0.03148em"}},"k")))),(0,n.yg)("span",{parentName:"span",className:"vlist-s"},"\u200b")),(0,n.yg)("span",{parentName:"span",className:"vlist-r"},(0,n.yg)("span",{parentName:"span",className:"vlist",style:{height:"0.15em"}},(0,n.yg)("span",{parentName:"span"})))))),(0,n.yg)("span",{parentName:"span",className:"mclose"},")"),(0,n.yg)("span",{parentName:"span",className:"mord"},".")))))),(0,n.yg)("p",null,"Given an initial guess ",(0,n.yg)("span",{parentName:"p",className:"math math-inline"},(0,n.yg)("span",{parentName:"span",className:"katex"},(0,n.yg)("span",{parentName:"span",className:"katex-mathml"},(0,n.yg)("math",{parentName:"span",xmlns:"http://www.w3.org/1998/Math/MathML"},(0,n.yg)("semantics",{parentName:"math"},(0,n.yg)("mrow",{parentName:"semantics"},(0,n.yg)("msub",{parentName:"mrow"},(0,n.yg)("mi",{parentName:"msub"},"x"),(0,n.yg)("mn",{parentName:"msub"},"0"))),(0,n.yg)("annotation",{parentName:"semantics",encoding:"application/x-tex"},"x_0")))),(0,n.yg)("span",{parentName:"span",className:"katex-html","aria-hidden":"true"},(0,n.yg)("span",{parentName:"span",className:"base"},(0,n.yg)("span",{parentName:"span",className:"strut",style:{height:"0.58056em",verticalAlign:"-0.15em"}}),(0,n.yg)("span",{parentName:"span",className:"mord"},(0,n.yg)("span",{parentName:"span",className:"mord mathnormal"},"x"),(0,n.yg)("span",{parentName:"span",className:"msupsub"},(0,n.yg)("span",{parentName:"span",className:"vlist-t vlist-t2"},(0,n.yg)("span",{parentName:"span",className:"vlist-r"},(0,n.yg)("span",{parentName:"span",className:"vlist",style:{height:"0.30110799999999993em"}},(0,n.yg)("span",{parentName:"span",style:{top:"-2.5500000000000003em",marginLeft:"0em",marginRight:"0.05em"}},(0,n.yg)("span",{parentName:"span",className:"pstrut",style:{height:"2.7em"}}),(0,n.yg)("span",{parentName:"span",className:"sizing reset-size6 size3 mtight"},(0,n.yg)("span",{parentName:"span",className:"mord mtight"},"0")))),(0,n.yg)("span",{parentName:"span",className:"vlist-s"},"\u200b")),(0,n.yg)("span",{parentName:"span",className:"vlist-r"},(0,n.yg)("span",{parentName:"span",className:"vlist",style:{height:"0.15em"}},(0,n.yg)("span",{parentName:"span"})))))))))),", we can write an optimizer using the steepest descent algorithm using the ",(0,n.yg)("strong",{parentName:"p"},"Optimizer()")," class in modOpt as follows:"),(0,n.yg)("pre",null,(0,n.yg)("code",{parentName:"pre",className:"language-py"},"import numpy as np\nimport time\n\nfrom modopt import Optimizer\n\n\nclass SteepestDescent(Optimizer):\n    def initialize(self):\n        # Name your algorithm\n        self.solver = 'steepest_descent'\n\n        self.obj = self.problem.compute_objective\n        self.grad = self.problem.compute_objective_gradient\n\n        self.options.declare('opt_tol', types=float)\n        # self.declare_outputs(x=2, f=1, opt=1, time=1)\n\n    def solve(self):\n        nx = self.problem.nx\n        x0 = x0 = self.problem.x.get_data()\n        opt_tol = self.options['opt_tol']\n        max_itr = self.options['max_itr']\n\n        obj = self.obj\n        grad = self.grad\n\n        start_time = time.time()\n\n        # Setting intial values for current iterates\n        x_k = x0 * 1.\n        f_k = obj(x_k)\n        g_k = grad(x_k)\n\n        itr = 0\n\n        opt = np.linalg.norm(g_k)\n\n        # Initializing outputs\n        self.update_outputs(itr=0,\n                            x=x0,\n                            obj=f_k,\n                            opt=opt,\n                            time=time.time() - start_time)\n\n        while (opt > opt_tol and itr < max_itr):\n            itr_start = time.time()\n            itr += 1\n\n            p_k = -g_k\n            x_k += p_k\n            f_k = obj(x_k)\n            g_k = grad(x_k)\n\n            opt = np.linalg.norm(g_k)\n\n            # Append outputs dict with new values from the current iteration\n            self.update_outputs(itr=itr,\n                                x=x_k,\n                                obj=f_k,\n                                opt=opt,\n                                time=time.time() - itr_start)\n\n        end_time = time.time()\n        self.total_time = end_time - start_time\n")),(0,n.yg)("p",null,"The ",(0,n.yg)("strong",{parentName:"p"},"Optimizer()")," class records all the data needed using the ",(0,n.yg)("inlineCode",{parentName:"p"},"outputs")," dictionary."))}o.isMDXComponent=!0}}]);