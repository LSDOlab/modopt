"use strict";(self.webpackChunkmy_website=self.webpackChunkmy_website||[]).push([[3730],{3905:function(t,e,r){r.d(e,{Zo:function(){return p},kt:function(){return f}});var n=r(7294);function i(t,e,r){return e in t?Object.defineProperty(t,e,{value:r,enumerable:!0,configurable:!0,writable:!0}):t[e]=r,t}function o(t,e){var r=Object.keys(t);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(t);e&&(n=n.filter((function(e){return Object.getOwnPropertyDescriptor(t,e).enumerable}))),r.push.apply(r,n)}return r}function a(t){for(var e=1;e<arguments.length;e++){var r=null!=arguments[e]?arguments[e]:{};e%2?o(Object(r),!0).forEach((function(e){i(t,e,r[e])})):Object.getOwnPropertyDescriptors?Object.defineProperties(t,Object.getOwnPropertyDescriptors(r)):o(Object(r)).forEach((function(e){Object.defineProperty(t,e,Object.getOwnPropertyDescriptor(r,e))}))}return t}function l(t,e){if(null==t)return{};var r,n,i=function(t,e){if(null==t)return{};var r,n,i={},o=Object.keys(t);for(n=0;n<o.length;n++)r=o[n],e.indexOf(r)>=0||(i[r]=t[r]);return i}(t,e);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(t);for(n=0;n<o.length;n++)r=o[n],e.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(t,r)&&(i[r]=t[r])}return i}var c=n.createContext({}),u=function(t){var e=n.useContext(c),r=e;return t&&(r="function"==typeof t?t(e):a(a({},e),t)),r},p=function(t){var e=u(t.components);return n.createElement(c.Provider,{value:e},t.children)},m={inlineCode:"code",wrapper:function(t){var e=t.children;return n.createElement(n.Fragment,{},e)}},s=n.forwardRef((function(t,e){var r=t.components,i=t.mdxType,o=t.originalType,c=t.parentName,p=l(t,["components","mdxType","originalType","parentName"]),s=u(r),f=i,d=s["".concat(c,".").concat(f)]||s[f]||m[f]||o;return r?n.createElement(d,a(a({ref:e},p),{},{components:r})):n.createElement(d,a({ref:e},p))}));function f(t,e){var r=arguments,i=e&&e.mdxType;if("string"==typeof t||i){var o=r.length,a=new Array(o);a[0]=s;var l={};for(var c in e)hasOwnProperty.call(e,c)&&(l[c]=e[c]);l.originalType=t,l.mdxType="string"==typeof t?t:i,a[1]=l;for(var u=2;u<o;u++)a[u]=r[u];return n.createElement.apply(null,a)}return n.createElement.apply(null,r)}s.displayName="MDXCreateElement"},7756:function(t,e,r){r.r(e),r.d(e,{frontMatter:function(){return l},contentTitle:function(){return c},metadata:function(){return u},toc:function(){return p},default:function(){return s}});var n=r(7462),i=r(3366),o=(r(7294),r(3905)),a=["components"],l={},c="Matrix-Matrix Multiplication",u={unversionedId:"std_lib_ref/Linear Algebra/matmat",id:"std_lib_ref/Linear Algebra/matmat",isDocsHomePage:!1,title:"Matrix-Matrix Multiplication",description:"This function allows you to compute matrix-matrix multiplication, as well as,",source:"@site/docs/std_lib_ref/Linear Algebra/matmat.mdx",sourceDirName:"std_lib_ref/Linear Algebra",slug:"/std_lib_ref/Linear Algebra/matmat",permalink:"/modopt/docs/std_lib_ref/Linear Algebra/matmat",editUrl:"https://github.com/lsdolab/modopt/edit/main/website/docs/std_lib_ref/Linear Algebra/matmat.mdx",tags:[],version:"current",frontMatter:{},sidebar:"docs",previous:{title:"Inner",permalink:"/modopt/docs/std_lib_ref/Linear Algebra/inner"},next:{title:"Matrix-Vector Multiplication",permalink:"/modopt/docs/std_lib_ref/Linear Algebra/matvec"}},p=[],m={toc:p};function s(t){var e=t.components,r=(0,i.Z)(t,a);return(0,o.kt)("wrapper",(0,n.Z)({},m,r,{components:e,mdxType:"MDXLayout"}),(0,o.kt)("h1",{id:"matrix-matrix-multiplication"},"Matrix-Matrix Multiplication"),(0,o.kt)("p",null,"This function allows you to compute matrix-matrix multiplication, as well as,\nmatrix-vector multiplication."),(0,o.kt)("p",null,"An example of how to use the operation is provided below."),(0,o.kt)("p",null,".. autofunction:: csdl.std.matmat.matmat"))}s.isMDXComponent=!0}}]);