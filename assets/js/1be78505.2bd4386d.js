"use strict";(self.webpackChunkmy_website=self.webpackChunkmy_website||[]).push([[656,364],{6886:(e,t,a)=>{a.r(t),a.d(t,{default:()=>U});var n=a(1504),c=a(5788),o=a(3315),l=a(6508),i=a(4971),r=a(3100),s=a(8072),d=a(3336),m=a(2416),u=a(5072);const b=e=>n.createElement("svg",(0,u.c)({width:"20",height:"20","aria-hidden":"true"},e),n.createElement("g",{fill:"#7a7a7a"},n.createElement("path",{d:"M9.992 10.023c0 .2-.062.399-.172.547l-4.996 7.492a.982.982 0 01-.828.454H1c-.55 0-1-.453-1-1 0-.2.059-.403.168-.551l4.629-6.942L.168 3.078A.939.939 0 010 2.528c0-.548.45-.997 1-.997h2.996c.352 0 .649.18.828.45L9.82 9.472c.11.148.172.347.172.55zm0 0"}),n.createElement("path",{d:"M19.98 10.023c0 .2-.058.399-.168.547l-4.996 7.492a.987.987 0 01-.828.454h-3c-.547 0-.996-.453-.996-1 0-.2.059-.403.168-.551l4.625-6.942-4.625-6.945a.939.939 0 01-.168-.55 1 1 0 01.996-.997h3c.348 0 .649.18.828.45l4.996 7.492c.11.148.168.347.168.55zm0 0"})));var p=a(8710),h=a(4724),E=a(8136),f=a(5756);const g={menuLinkText:"menuLinkText_OKON"},_=(e,t)=>"link"===e.type?(0,r.Sc)(e.href,t):"category"===e.type&&e.items.some((e=>_(e,t))),v=(0,n.memo)((function(e){let{items:t,...a}=e;return n.createElement(n.Fragment,null,t.map(((e,t)=>n.createElement(C,(0,u.c)({key:t,item:e},a)))))}));function C(e){let{item:t,...a}=e;return"category"===t.type?0===t.items.length?null:n.createElement(k,(0,u.c)({item:t},a)):n.createElement(S,(0,u.c)({item:t},a))}function k(e){let{item:t,onItemClick:a,activePath:c,...o}=e;const{items:l,label:s,collapsible:d}=t,m=_(t,c),{collapsed:b,setCollapsed:p,toggleCollapsed:h}=(0,r.au)({initialState:()=>!!d&&(!m&&t.collapsed)});return function(e){let{isActive:t,collapsed:a,setCollapsed:c}=e;const o=(0,r.i0)(t);(0,n.useEffect)((()=>{t&&!o&&a&&c(!1)}),[t,o,a])}({isActive:m,collapsed:b,setCollapsed:p}),n.createElement("li",{className:(0,i.c)(r.Wu.docs.docSidebarItemCategory,"menu__list-item",{"menu__list-item--collapsed":b})},n.createElement("a",(0,u.c)({className:(0,i.c)("menu__link",{"menu__link--sublist":d,"menu__link--active":d&&m,[g.menuLinkText]:!d}),onClick:d?e=>{e.preventDefault(),h()}:void 0,href:d?"#":void 0},o),s),n.createElement(r.Uv,{lazy:!0,as:"ul",className:"menu__list",collapsed:b},n.createElement(v,{items:l,tabIndex:b?-1:0,onItemClick:a,activePath:c})))}function S(e){let{item:t,onItemClick:a,activePath:c,...o}=e;const{href:l,label:s}=t,d=_(t,c);return n.createElement("li",{className:(0,i.c)(r.Wu.docs.docSidebarItemLink,"menu__list-item"),key:s},n.createElement(h.c,(0,u.c)({className:(0,i.c)("menu__link",{"menu__link--active":d}),"aria-current":d?"page":void 0,to:l},(0,E.c)(l)&&{onClick:a},o),(0,E.c)(l)?s:n.createElement("span",null,s,n.createElement(f.c,null))))}const N={sidebar:"sidebar_a3j0",sidebarWithHideableNavbar:"sidebarWithHideableNavbar_VlPv",sidebarHidden:"sidebarHidden_OqfG",sidebarLogo:"sidebarLogo_hmkv",menu:"menu_cyFh",menuWithAnnouncementBar:"menuWithAnnouncementBar_+O1J",collapseSidebarButton:"collapseSidebarButton_eoK2",collapseSidebarButtonIcon:"collapseSidebarButtonIcon_e+kA",sidebarMenuIcon:"sidebarMenuIcon_iZzd",sidebarMenuCloseIcon:"sidebarMenuCloseIcon_6kU2"};function I(e){let{onClick:t}=e;return n.createElement("button",{type:"button",title:(0,p.G)({id:"theme.docs.sidebar.collapseButtonTitle",message:"Collapse sidebar",description:"The title attribute for collapse button of doc sidebar"}),"aria-label":(0,p.G)({id:"theme.docs.sidebar.collapseButtonAriaLabel",message:"Collapse sidebar",description:"The title attribute for collapse button of doc sidebar"}),className:(0,i.c)("button button--secondary button--outline",N.collapseSidebarButton),onClick:t},n.createElement(b,{className:N.collapseSidebarButtonIcon}))}function T(e){let{path:t,sidebar:a,onCollapse:c,isHidden:o}=e;const l=function(){const{isClosed:e}=(0,r.el)(),[t,a]=(0,n.useState)(!e);return(0,d.c)((t=>{let{scrollY:n}=t;e||a(0===n)})),t}(),{navbar:{hideOnScroll:s},hideableSidebar:u}=(0,r.yw)(),{isClosed:b}=(0,r.el)();return n.createElement("div",{className:(0,i.c)(N.sidebar,{[N.sidebarWithHideableNavbar]:s,[N.sidebarHidden]:o})},s&&n.createElement(m.c,{tabIndex:-1,className:N.sidebarLogo}),n.createElement("nav",{className:(0,i.c)("menu thin-scrollbar",N.menu,{[N.menuWithAnnouncementBar]:!b&&l})},n.createElement("ul",{className:(0,i.c)(r.Wu.docs.docSidebarMenu,"menu__list")},n.createElement(v,{items:a,activePath:t}))),u&&n.createElement(I,{onClick:c}))}const w=e=>{let{toggleSidebar:t,sidebar:a,path:c}=e;return n.createElement("ul",{className:(0,i.c)(r.Wu.docs.docSidebarMenu,"menu__list")},n.createElement(v,{items:a,activePath:c,onItemClick:()=>t()}))};function y(e){return n.createElement(r.oh,{component:w,props:e})}const M=n.memo(T),B=n.memo(y);function x(e){const t=(0,s.c)(),a="desktop"===t||"ssr"===t,c="mobile"===t;return n.createElement(n.Fragment,null,a&&n.createElement(M,e),c&&n.createElement(B,e))}var W=a(1036),A=a(6364),H=a(5592);const L="backToTopButton_i9tI",P="backToTopButtonShow_wCmF",F=!1;function D(){const e=(0,n.useRef)(null);return{smoothScrollTop:function(){e.current=F?(window.scrollTo({top:0,behavior:"smooth"}),()=>{}):function(){let e=null;return function t(){const a=document.documentElement.scrollTop;a>0&&(e=requestAnimationFrame(t),window.scrollTo(0,Math.floor(.85*a)))}(),()=>e&&cancelAnimationFrame(e)}()},cancelScrollToTop:()=>e.current?.()}}const R=function(){const e=(0,H.IT)(),{smoothScrollTop:t,cancelScrollToTop:a}=D(),[c,o]=(0,n.useState)(!1);return(0,d.c)(((e,t)=>{let{scrollY:n}=e;if(!t)return;const c=n<t.scrollY;if(c||a(),n<300)o(!1);else if(c){const e=document.documentElement.scrollHeight;n+window.innerHeight<e&&o(!0)}else o(!1)}),[e]),n.createElement("button",{className:(0,i.c)("clean-btn",L,{[P]:c}),type:"button",onClick:()=>t()},n.createElement("svg",{viewBox:"0 0 24 24",width:"28"},n.createElement("path",{d:"M7.41 15.41L12 10.83l4.59 4.58L18 14l-6-6-6 6z",fill:"currentColor"})))},z={docPage:"docPage_lDyR",docMainContainer:"docMainContainer_r8cw",docSidebarContainer:"docSidebarContainer_0YBq",docMainContainerEnhanced:"docMainContainerEnhanced_SOUu",docSidebarContainerHidden:"docSidebarContainerHidden_Qlt2",collapsedDocSidebar:"collapsedDocSidebar_zZpm",expandSidebarButtonIcon:"expandSidebarButtonIcon_cxi8",docItemWrapperEnhanced:"docItemWrapperEnhanced_aT5H"};var G=a(2956);function O(e){let{currentDocRoute:t,versionMetadata:a,children:o}=e;const{pluginId:s,version:d}=a,m=t.sidebar,u=m?a.docsSidebars[m]:void 0,[h,E]=(0,n.useState)(!1),[f,g]=(0,n.useState)(!1),_=(0,n.useCallback)((()=>{f&&g(!1),E(!h)}),[f]);return n.createElement(l.c,{wrapperClassName:r.Wu.wrapper.docsPages,pageClassName:r.Wu.page.docsDocPage,searchMetadatas:{version:d,tag:(0,r.SE)(s,d)}},n.createElement("div",{className:z.docPage},n.createElement(R,null),u&&n.createElement("aside",{className:(0,i.c)(z.docSidebarContainer,{[z.docSidebarContainerHidden]:h}),onTransitionEnd:e=>{e.currentTarget.classList.contains(z.docSidebarContainer)&&h&&g(!0)}},n.createElement(x,{key:m,sidebar:u,path:t.path,onCollapse:_,isHidden:f}),f&&n.createElement("div",{className:z.collapsedDocSidebar,title:(0,p.G)({id:"theme.docs.sidebar.expandButtonTitle",message:"Expand sidebar",description:"The ARIA label and title attribute for expand button of doc sidebar"}),"aria-label":(0,p.G)({id:"theme.docs.sidebar.expandButtonAriaLabel",message:"Expand sidebar",description:"The ARIA label and title attribute for expand button of doc sidebar"}),tabIndex:0,role:"button",onKeyDown:_,onClick:_},n.createElement(b,{className:z.expandSidebarButtonIcon}))),n.createElement("main",{className:(0,i.c)(z.docMainContainer,{[z.docMainContainerEnhanced]:h||!u})},n.createElement("div",{className:(0,i.c)("container padding-top--md padding-bottom--lg",z.docItemWrapper,{[z.docItemWrapperEnhanced]:h})},n.createElement(c.Iu,{components:W.c},o)))))}const U=function(e){const{route:{routes:t},versionMetadata:a,location:c}=e,l=t.find((e=>(0,H.ot)(c.pathname,e)));return l?n.createElement(n.Fragment,null,n.createElement(G.c,null,n.createElement("html",{className:a.className})),n.createElement(O,{currentDocRoute:l,versionMetadata:a},(0,o.c)(t,{versionMetadata:a}))):n.createElement(A.default,e)}},6364:(e,t,a)=>{a.r(t),a.d(t,{default:()=>l});var n=a(1504),c=a(6508),o=a(8710);const l=function(){return n.createElement(c.c,{title:(0,o.G)({id:"theme.NotFound.title",message:"Page Not Found"})},n.createElement("main",{className:"container margin-vert--xl"},n.createElement("div",{className:"row"},n.createElement("div",{className:"col col--6 col--offset-3"},n.createElement("h1",{className:"hero__title"},n.createElement(o.c,{id:"theme.NotFound.title",description:"The title of the 404 page"},"Page Not Found")),n.createElement("p",null,n.createElement(o.c,{id:"theme.NotFound.p1",description:"The first paragraph of the 404 page"},"We could not find what you were looking for.")),n.createElement("p",null,n.createElement(o.c,{id:"theme.NotFound.p2",description:"The 2nd paragraph of the 404 page"},"Please contact the owner of the site that linked you to the original URL and let them know their link is broken."))))))}}}]);