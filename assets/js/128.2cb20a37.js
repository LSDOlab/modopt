"use strict";(self.webpackChunkmy_website=self.webpackChunkmy_website||[]).push([[128],{2944:(e,t,n)=>{n.d(t,{c:()=>a});const a=()=>null},5756:(e,t,n)=>{n.d(t,{c:()=>r});var a=n(1504);const c="iconExternalLink_wgqa",r=e=>{let{width:t=13.5,height:n=13.5}=e;return a.createElement("svg",{width:t,height:n,"aria-hidden":"true",viewBox:"0 0 24 24",className:c},a.createElement("path",{fill:"currentColor",d:"M21 13v10h-21v-19h12v2h-10v15h17v-8h2zm3-12h-10.988l4.035 4-6.977 7.07 2.828 2.828 6.977-7.07 4.125 4.172v-11z"}))}},6508:(e,t,n)=>{n.d(t,{c:()=>Ee});var a=n(1504),c=n(4971),r=n(5592),l=n(8710),o=n(3100);const s="skipToContent_OuoZ";function i(e){e.setAttribute("tabindex","-1"),e.focus(),e.removeAttribute("tabindex")}const m=function(){const e=(0,a.useRef)(null),{action:t}=(0,r.Uz)();return(0,o.cD)((n=>{let{location:a}=n;e.current&&!a.hash&&"PUSH"===t&&i(e.current)})),a.createElement("div",{ref:e},a.createElement("a",{href:"#",className:s,onClick:e=>{e.preventDefault();const t=document.querySelector("main:first-of-type")||document.querySelector(".main-wrapper");t&&i(t)}},a.createElement(l.c,{id:"theme.common.skipToMainContent",description:"The skip to content label used for accessibility, allowing to rapidly navigate to main content with keyboard tab/enter navigation"},"Skip to main content")))};var d=n(5072);function u(e){let{width:t=20,height:n=20,className:c,...r}=e;return a.createElement("svg",(0,d.c)({className:c,viewBox:"0 0 24 24",width:t,height:n,fill:"currentColor"},r),a.createElement("path",{d:"M24 20.188l-8.315-8.209 8.2-8.282-3.697-3.697-8.212 8.318-8.31-8.203-3.666 3.666 8.321 8.24-8.206 8.313 3.666 3.666 8.237-8.318 8.285 8.203z"}))}const h="announcementBar_axC9",b="announcementBarPlaceholder_xYHE",g="announcementBarClose_A3A1",v="announcementBarContent_6uhP";const f=function(){const{isClosed:e,close:t}=(0,o.el)(),{announcementBar:n}=(0,o.yw)();if(!n)return null;const{content:r,backgroundColor:s,textColor:i,isCloseable:m}=n;return!r||m&&e?null:a.createElement("div",{className:h,style:{backgroundColor:s,color:i},role:"banner"},m&&a.createElement("div",{className:b}),a.createElement("div",{className:v,dangerouslySetInnerHTML:{__html:r}}),m?a.createElement("button",{type:"button",className:(0,c.c)("clean-btn close",g),onClick:t,"aria-label":(0,l.G)({id:"theme.AnnouncementBar.closeButtonAriaLabel",message:"Close",description:"The ARIA label for close button of announcement bar"})},a.createElement(u,{width:14,height:14})):null)};var E=n(2944),p=n(3664);const k={toggle:"toggle_iYfV"},_=e=>{let{icon:t,style:n}=e;return a.createElement("span",{className:(0,c.c)(k.toggle,k.dark),style:n},t)},w=e=>{let{icon:t,style:n}=e;return a.createElement("span",{className:(0,c.c)(k.toggle,k.light),style:n},t)},N=(0,a.memo)((e=>{let{className:t,icons:n,checked:r,disabled:l,onChange:o}=e;const[s,i]=(0,a.useState)(r),[m,d]=(0,a.useState)(!1),u=(0,a.useRef)(null);return a.createElement("div",{className:(0,c.c)("react-toggle",t,{"react-toggle--checked":s,"react-toggle--focus":m,"react-toggle--disabled":l})},a.createElement("div",{className:"react-toggle-track",role:"button",tabIndex:-1,onClick:()=>u.current?.click()},a.createElement("div",{className:"react-toggle-track-check"},n.checked),a.createElement("div",{className:"react-toggle-track-x"},n.unchecked),a.createElement("div",{className:"react-toggle-thumb"})),a.createElement("input",{ref:u,checked:s,type:"checkbox",className:"react-toggle-screenreader-only","aria-label":"Switch between dark and light mode",onChange:o,onClick:()=>i(!s),onFocus:()=>d(!0),onBlur:()=>d(!1),onKeyDown:e=>{"Enter"===e.key&&u.current?.click()}}))}));function y(e){const{colorMode:{switchConfig:{darkIcon:t,darkIconStyle:n,lightIcon:c,lightIconStyle:r}}}=(0,o.yw)(),l=(0,p.c)();return a.createElement(N,(0,d.c)({disabled:!l,icons:{checked:a.createElement(_,{icon:t,style:n}),unchecked:a.createElement(w,{icon:c,style:r})}},e))}var C=n(9464),S=n(3336);const D=e=>{const t=(0,r.IT)(),[n,c]=(0,a.useState)(e),l=(0,a.useRef)(!1),[s,i]=(0,a.useState)(0),m=(0,a.useCallback)((e=>{null!==e&&i(e.getBoundingClientRect().height)}),[]);return(0,S.c)(((t,n)=>{const a=t.scrollY,r=n?.scrollY;if(!e)return;if(a<s)return void c(!0);if(l.current)return l.current=!1,void c(!1);r&&0===a&&c(!0);const o=document.documentElement.scrollHeight-s,i=window.innerHeight;r&&a>=r?c(!1):a+i<o&&c(!0)}),[s,l]),(0,o.cD)((t=>{e&&!t.location.hash&&c(!0)})),(0,a.useEffect)((()=>{e&&t.hash&&(l.current=!0)}),[t.hash]),{navbarRef:m,isNavbarVisible:n}};const I=function(e){void 0===e&&(e=!0),(0,a.useEffect)((()=>(document.body.style.overflow=e?"hidden":"visible",()=>{document.body.style.overflow="visible"})),[e])};var T=n(8072),L=n(3920),B=n(5384),x=n(2416);const A=e=>{let{width:t=30,height:n=30,className:c,...r}=e;return a.createElement("svg",(0,d.c)({className:c,width:t,height:n,viewBox:"0 0 30 30","aria-hidden":"true"},r),a.createElement("path",{stroke:"currentColor",strokeLinecap:"round",strokeMiterlimit:"10",strokeWidth:"2",d:"M4 7h22M4 15h22M4 23h22"}))};function M(e){let{width:t=20,height:n=20,className:c,...r}=e;return a.createElement("svg",(0,d.c)({className:c,viewBox:"0 0 413.348 413.348",width:t,height:n,fill:"currentColor"},r),a.createElement("path",{d:"m413.348 24.354-24.354-24.354-182.32 182.32-182.32-182.32-24.354 24.354 182.32 182.32-182.32 182.32 24.354 24.354 182.32-182.32 182.32 182.32 24.354-24.354-182.32-182.32z"}))}const R={toggle:"toggle_2i4l",navbarHideable:"navbarHideable_RReh",navbarHidden:"navbarHidden_FBwS",navbarSidebarToggle:"navbarSidebarToggle_AVbO",navbarSidebarCloseSvg:"navbarSidebarCloseSvg_+9jJ"},P="right";function H(){return(0,o.yw)().navbar.items}function V(){const{colorMode:{disableSwitch:e}}=(0,o.yw)(),{isDarkTheme:t,setLightTheme:n,setDarkTheme:c}=(0,C.c)();return{isDarkTheme:t,toggle:(0,a.useCallback)((e=>e.target.checked?c():n()),[n,c]),disabled:e}}function U(e){let{sidebarShown:t,toggleSidebar:n}=e;I(t);const r=H(),s=V(),i=function(e){let{sidebarShown:t,toggleSidebar:n}=e;const c=(0,o.GW)()?.({toggleSidebar:n}),r=(0,o.i0)(c),[l,s]=(0,a.useState)((()=>!1));(0,a.useEffect)((()=>{c&&!r&&s(!0)}),[c,r]);const i=!!c;return(0,a.useEffect)((()=>{i?t||s(!0):s(!1)}),[t,i]),{shown:l,hide:(0,a.useCallback)((()=>{s(!1)}),[]),content:c}}({sidebarShown:t,toggleSidebar:n});return a.createElement("div",{className:"navbar-sidebar"},a.createElement("div",{className:"navbar-sidebar__brand"},a.createElement(x.c,{className:"navbar__brand",imageClassName:"navbar__logo",titleClassName:"navbar__title"}),!s.disabled&&a.createElement(y,{className:R.navbarSidebarToggle,checked:s.isDarkTheme,onChange:s.toggle}),a.createElement("button",{type:"button",className:"clean-btn navbar-sidebar__close",onClick:n},a.createElement(M,{width:20,height:20,className:R.navbarSidebarCloseSvg}))),a.createElement("div",{className:(0,c.c)("navbar-sidebar__items",{"navbar-sidebar__items--show-secondary":i.shown})},a.createElement("div",{className:"navbar-sidebar__item menu"},a.createElement("ul",{className:"menu__list"},r.map(((e,t)=>a.createElement(B.c,(0,d.c)({mobile:!0},e,{onClick:n,key:t})))))),a.createElement("div",{className:"navbar-sidebar__item menu"},r.length>0&&a.createElement("button",{type:"button",className:"clean-btn navbar-sidebar__back",onClick:i.hide},a.createElement(l.c,{id:"theme.navbar.mobileSidebarSecondaryMenu.backButtonLabel",description:"The label of the back button to return to main menu, inside the mobile navbar sidebar secondary menu (notably used to display the docs sidebar)"},"\u2190 Back to main menu")),i.content)))}const z=function(){const{navbar:{hideOnScroll:e,style:t}}=(0,o.yw)(),n=function(){const e=(0,T.c)(),t="mobile"===e,[n,c]=(0,a.useState)(!1);(0,o.a4)((()=>{n&&c(!1)}));const r=(0,a.useCallback)((()=>{c((e=>!e))}),[]);return(0,a.useEffect)((()=>{"desktop"===e&&c(!1)}),[e]),{shouldRender:t,toggle:r,shown:n}}(),r=V(),l=(0,L.UF)(),{navbarRef:s,isNavbarVisible:i}=D(e),m=H(),u=m.some((e=>"search"===e.type)),{leftItems:h,rightItems:b}=function(e){return{leftItems:e.filter((e=>"left"===(e.position??P))),rightItems:e.filter((e=>"right"===(e.position??P)))}}(m);return a.createElement("nav",{ref:s,className:(0,c.c)("navbar","navbar--fixed-top",{"navbar--dark":"dark"===t,"navbar--primary":"primary"===t,"navbar-sidebar--show":n.shown,[R.navbarHideable]:e,[R.navbarHidden]:e&&!i})},a.createElement("div",{className:"navbar__inner"},a.createElement("div",{className:"navbar__items"},(m?.length>0||l)&&a.createElement("button",{"aria-label":"Navigation bar toggle",className:"navbar__toggle clean-btn",type:"button",tabIndex:0,onClick:n.toggle,onKeyDown:n.toggle},a.createElement(A,null)),a.createElement(x.c,{className:"navbar__brand",imageClassName:"navbar__logo",titleClassName:"navbar__title"}),h.map(((e,t)=>a.createElement(B.c,(0,d.c)({},e,{key:t}))))),a.createElement("div",{className:"navbar__items navbar__items--right"},b.map(((e,t)=>a.createElement(B.c,(0,d.c)({},e,{key:t})))),!r.disabled&&a.createElement(y,{className:R.toggle,checked:r.isDarkTheme,onChange:r.toggle}),!u&&a.createElement(E.c,null))),a.createElement("div",{role:"presentation",className:"navbar-sidebar__backdrop",onClick:n.toggle}),n.shouldRender&&a.createElement(U,{sidebarShown:n.shown,toggleSidebar:n.toggle}))};var G=n(4724),$=n(964),O=n(8136);const F="footerLogoLink_SRtH";var W=n(280),Y=n(5756);function q(e){let{to:t,href:n,label:c,prependBaseUrlToHref:r,...l}=e;const o=(0,$.c)(t),s=(0,$.c)(n,{forcePrependBaseUrl:!0});return a.createElement(G.c,(0,d.c)({className:"footer__link-item"},n?{href:r?s:n}:{to:o},l),n&&!(0,O.c)(n)?a.createElement("span",null,c,a.createElement(Y.c,null)):c)}const K=e=>{let{sources:t,alt:n}=e;return a.createElement(W.c,{className:"footer__logo",alt:n,sources:t})};const j=function(){const{footer:e}=(0,o.yw)(),{copyright:t,links:n=[],logo:r={}}=e||{},l={light:(0,$.c)(r.src),dark:(0,$.c)(r.srcDark||r.src)};return e?a.createElement("footer",{className:(0,c.c)("footer",{"footer--dark":"dark"===e.style})},a.createElement("div",{className:"container"},n&&n.length>0&&a.createElement("div",{className:"row footer__links"},n.map(((e,t)=>a.createElement("div",{key:t,className:"col footer__col"},null!=e.title?a.createElement("div",{className:"footer__title"},e.title):null,null!=e.items&&Array.isArray(e.items)&&e.items.length>0?a.createElement("ul",{className:"footer__items"},e.items.map(((e,t)=>e.html?a.createElement("li",{key:t,className:"footer__item",dangerouslySetInnerHTML:{__html:e.html}}):a.createElement("li",{key:e.href||e.to,className:"footer__item"},a.createElement(q,e))))):null)))),(r||t)&&a.createElement("div",{className:"footer__bottom text--center"},r&&(r.src||r.srcDark)&&a.createElement("div",{className:"margin-bottom--sm"},r.href?a.createElement(G.c,{href:r.href,className:F},a.createElement(K,{alt:r.alt,sources:l})):a.createElement(K,{alt:r.alt,sources:l})),t?a.createElement("div",{className:"footer__copyright",dangerouslySetInnerHTML:{__html:t}}):null))):null};var Q=n(8684);const X=(0,o.GS)("theme"),Z="light",J="dark",ee=e=>e===J?J:Z,te=e=>{(0,o.GS)("theme").set(ee(e))},ne=()=>{const{colorMode:{defaultMode:e,disableSwitch:t,respectPrefersColorScheme:n}}=(0,o.yw)(),[c,r]=(0,a.useState)((e=>Q.c.canUseDOM?ee(document.documentElement.getAttribute("data-theme")):ee(e))(e)),l=(0,a.useCallback)((()=>{r(Z),te(Z)}),[]),s=(0,a.useCallback)((()=>{r(J),te(J)}),[]);return(0,a.useEffect)((()=>{document.documentElement.setAttribute("data-theme",ee(c))}),[c]),(0,a.useEffect)((()=>{if(!t)try{const e=X.get();null!==e&&r(ee(e))}catch(e){console.error(e)}}),[r]),(0,a.useEffect)((()=>{t&&!n||window.matchMedia("(prefers-color-scheme: dark)").addListener((e=>{let{matches:t}=e;r(t?J:Z)}))}),[]),{isDarkTheme:c===J,setLightTheme:l,setDarkTheme:s}};var ae=n(520);const ce=function(e){const{isDarkTheme:t,setLightTheme:n,setDarkTheme:c}=ne();return a.createElement(ae.c.Provider,{value:{isDarkTheme:t,setLightTheme:n,setDarkTheme:c}},e.children)},re="docusaurus.tab.",le=()=>{const[e,t]=(0,a.useState)({}),n=(0,a.useCallback)(((e,t)=>{(0,o.GS)(`${re}${e}`).set(t)}),[]);return(0,a.useEffect)((()=>{try{const e={};(0,o.Ed)().forEach((t=>{if(t.startsWith(re)){const n=t.substring(15);e[n]=(0,o.GS)(t).get()}})),t(e)}catch(e){console.error(e)}}),[]),{tabGroupChoices:e,setTabGroupChoices:(e,a)=>{t((t=>({...t,[e]:a}))),n(e,a)}}},oe=(0,a.createContext)(void 0);const se=function(e){const{tabGroupChoices:t,setTabGroupChoices:n}=le();return a.createElement(oe.Provider,{value:{tabGroupChoices:t,setTabGroupChoices:n}},e.children)};function ie(e){let{children:t}=e;return a.createElement(ce,null,a.createElement(o.qu,null,a.createElement(se,null,a.createElement(o.gc,null,a.createElement(o.cz,null,t)))))}var me=n(2956),de=n(8264);function ue(e){let{locale:t,version:n,tag:c}=e;return a.createElement(me.c,null,t&&a.createElement("meta",{name:"docusaurus_locale",content:t}),n&&a.createElement("meta",{name:"docusaurus_version",content:n}),c&&a.createElement("meta",{name:"docusaurus_tag",content:c}))}var he=n(6068);function be(){const{i18n:{defaultLocale:e,locales:t}}=(0,de.c)(),n=(0,o.DP)();return a.createElement(me.c,null,t.map((e=>a.createElement("link",{key:e,rel:"alternate",href:n.createUrl({locale:e,fullyQualified:!0}),hrefLang:e}))),a.createElement("link",{rel:"alternate",href:n.createUrl({locale:e,fullyQualified:!0}),hrefLang:"x-default"}))}function ge(e){let{permalink:t}=e;const{siteConfig:{url:n}}=(0,de.c)(),c=function(){const{siteConfig:{url:e}}=(0,de.c)(),{pathname:t}=(0,r.IT)();return e+(0,$.c)(t)}(),l=t?`${n}${t}`:c;return a.createElement(me.c,null,a.createElement("meta",{property:"og:url",content:l}),a.createElement("link",{rel:"canonical",href:l}))}function ve(e){const{siteConfig:{favicon:t},i18n:{currentLocale:n,localeConfigs:c}}=(0,de.c)(),{metadatas:r,image:l}=(0,o.yw)(),{title:s,description:i,image:m,keywords:u,searchMetadatas:h}=e,b=(0,$.c)(t),g=(0,o.g7)(s),v=n,f=c[n].direction;return a.createElement(a.Fragment,null,a.createElement(me.c,null,a.createElement("html",{lang:v,dir:f}),t&&a.createElement("link",{rel:"shortcut icon",href:b}),a.createElement("title",null,g),a.createElement("meta",{property:"og:title",content:g}),a.createElement("meta",{name:"twitter:card",content:"summary_large_image"})),l&&a.createElement(he.c,{image:l}),m&&a.createElement(he.c,{image:m}),a.createElement(he.c,{description:i,keywords:u}),a.createElement(ge,null),a.createElement(be,null),a.createElement(ue,(0,d.c)({tag:o.e6,locale:n},h)),a.createElement(me.c,null,r.map(((e,t)=>a.createElement("meta",(0,d.c)({key:`metadata_${t}`},e))))))}const fe=function(){(0,a.useEffect)((()=>{const e="navigation-with-keyboard";function t(t){"keydown"===t.type&&"Tab"===t.key&&document.body.classList.add(e),"mousedown"===t.type&&document.body.classList.remove(e)}return document.addEventListener("keydown",t),document.addEventListener("mousedown",t),()=>{document.body.classList.remove(e),document.removeEventListener("keydown",t),document.removeEventListener("mousedown",t)}}),[])};const Ee=function(e){const{children:t,noFooter:n,wrapperClassName:r,pageClassName:l}=e;return fe(),a.createElement(ie,null,a.createElement(ve,e),a.createElement(m,null),a.createElement(f,null),a.createElement(z,null),a.createElement("div",{className:(0,c.c)(o.Wu.wrapper.main,r,l)},t),!n&&a.createElement(j,null))}},2416:(e,t,n)=>{n.d(t,{c:()=>m});var a=n(5072),c=n(1504),r=n(4724),l=n(280),o=n(964),s=n(8264),i=n(3100);const m=e=>{const{siteConfig:{title:t}}=(0,s.c)(),{navbar:{title:n,logo:m={src:""}}}=(0,i.yw)(),{imageClassName:d,titleClassName:u,...h}=e,b=(0,o.c)(m.href||"/"),g={light:(0,o.c)(m.src),dark:(0,o.c)(m.srcDark||m.src)};return c.createElement(r.c,(0,a.c)({to:b},h,m.target&&{target:m.target}),m.src&&c.createElement(l.c,{className:d,sources:g,alt:m.alt||n||t}),null!=n&&c.createElement("b",{className:u},n))}},9428:(e,t,n)=>{n.d(t,{A:()=>u,c:()=>g});var a=n(5072),c=n(1504),r=n(4971),l=n(4724),o=n(964),s=n(5756),i=n(8136),m=n(5384);const d="dropdown__link--active";function u(e){let{activeBasePath:t,activeBaseRegex:n,to:r,href:m,label:u,activeClassName:h="",prependBaseUrlToHref:b,...g}=e;const v=(0,o.c)(r),f=(0,o.c)(t),E=(0,o.c)(m,{forcePrependBaseUrl:!0}),p=u&&m&&!(0,i.c)(m),k=h===d;return c.createElement(l.c,(0,a.c)({},m?{href:b?E:m}:{isNavLink:!0,activeClassName:g.className?.includes(h)?"":h,to:v,...t||n?{isActive:(e,t)=>n?new RegExp(n).test(t.pathname):t.pathname.startsWith(f)}:null},g),p?c.createElement("span",null,u,c.createElement(s.c,k&&{width:12,height:12})):u)}function h(e){let{className:t,isDropdownItem:n=!1,...l}=e;const o=c.createElement(u,(0,a.c)({className:(0,r.c)(n?"dropdown__link":"navbar__item navbar__link",t)},l));return n?c.createElement("li",null,o):o}function b(e){let{className:t,isDropdownItem:n,...l}=e;return c.createElement("li",{className:"menu__list-item"},c.createElement(u,(0,a.c)({className:(0,r.c)("menu__link",t)},l)))}const g=function(e){let{mobile:t=!1,position:n,...r}=e;const l=t?b:h;return c.createElement(l,(0,a.c)({},r,{activeClassName:r.activeClassName??(0,m.e)(t)}))}},3040:(e,t,n)=>{n.d(t,{c:()=>d});var a=n(5072),c=n(1504),r=n(9428),l=n(3920),o=n(4971),s=n(5384),i=n(3100),m=n(5684);function d(e){let{docId:t,label:n,docsPluginId:d,...u}=e;const{activeVersion:h,activeDoc:b}=(0,l.wB)(d),{preferredVersion:g}=(0,i.iy)(d),v=(0,l.aA)(d),f=function(e,t){const n=e.flatMap((e=>e.docs)),a=n.find((e=>e.id===t));if(!a){const a=n.map((e=>e.id)).join("\n- ");throw new Error(`DocNavbarItem: couldn't find any doc with id "${t}" in version${e.length?"s":""} ${e.map((e=>e.name)).join(", ")}".\nAvailable doc ids are:\n- ${a}`)}return a}((0,m.uniq)([h,g,v].filter(Boolean)),t),E=(0,s.e)(u.mobile);return c.createElement(r.c,(0,a.c)({exact:!0},u,{className:(0,o.c)(u.className,{[E]:b?.sidebar&&b.sidebar===f.sidebar}),activeClassName:E,label:n??f.id,to:f.path}))}},5692:(e,t,n)=>{n.d(t,{c:()=>d});var a=n(5072),c=n(1504),r=n(9428),l=n(40),o=n(3920),s=n(3100),i=n(8710);const m=e=>e.docs.find((t=>t.id===e.mainDocId));function d(e){let{mobile:t,docsPluginId:n,dropdownActiveClassDisabled:d,dropdownItemsBefore:u,dropdownItemsAfter:h,...b}=e;const g=(0,o.wB)(n),v=(0,o.gN)(n),f=(0,o.aA)(n),{preferredVersion:E,savePreferredVersionName:p}=(0,s.iy)(n);const k=function(){const e=v.map((e=>{const t=g?.alternateDocVersions[e.name]||m(e);return{isNavLink:!0,label:e.label,to:t.path,isActive:()=>e===g?.activeVersion,onClick:()=>{p(e.name)}}}));return[...u,...e,...h]}(),_=g.activeVersion??E??f,w=t&&k?(0,i.G)({id:"theme.navbar.mobileVersionsDropdown.label",message:"Versions",description:"The label for the navbar versions dropdown on mobile view"}):_.label,N=t&&k?void 0:m(_).path;return k.length<=1?c.createElement(r.c,(0,a.c)({},b,{mobile:t,label:w,to:N,isActive:d?()=>!1:void 0})):c.createElement(l.c,(0,a.c)({},b,{mobile:t,label:w,to:N,items:k,isActive:d?()=>!1:void 0}))}},4168:(e,t,n)=>{n.d(t,{c:()=>i});var a=n(5072),c=n(1504),r=n(9428),l=n(3920),o=n(3100);const s=e=>e.docs.find((t=>t.id===e.mainDocId));function i(e){let{label:t,to:n,docsPluginId:i,...m}=e;const d=(0,l.MK)(i),{preferredVersion:u}=(0,o.iy)(i),h=(0,l.aA)(i),b=d??u??h,g=t??b.label,v=n??s(b).path;return c.createElement(r.c,(0,a.c)({},m,{label:g,to:v}))}},40:(e,t,n)=>{n.d(t,{c:()=>h});var a=n(5072),c=n(1504),r=n(4971),l=n(3100),o=n(9428),s=n(5384);const i="dropdown__link--active";function m(e,t){return e.some((e=>function(e,t){return!!(0,l.Sc)(e.to,t)||!(!e.activeBaseRegex||!new RegExp(e.activeBaseRegex).test(t))||!(!e.activeBasePath||!t.startsWith(e.activeBasePath))}(e,t)))}function d(e){let{items:t,position:n,className:l,...m}=e;const d=(0,c.useRef)(null),u=(0,c.useRef)(null),[h,b]=(0,c.useState)(!1);return(0,c.useEffect)((()=>{const e=e=>{d.current&&!d.current.contains(e.target)&&b(!1)};return document.addEventListener("mousedown",e),document.addEventListener("touchstart",e),()=>{document.removeEventListener("mousedown",e),document.removeEventListener("touchstart",e)}}),[d]),c.createElement("div",{ref:d,className:(0,r.c)("navbar__item","dropdown","dropdown--hoverable",{"dropdown--right":"right"===n,"dropdown--show":h})},c.createElement(o.A,(0,a.c)({className:(0,r.c)("navbar__link",l)},m,{onClick:m.to?void 0:e=>e.preventDefault(),onKeyDown:e=>{"Enter"===e.key&&(e.preventDefault(),b(!h))}}),m.children??m.label),c.createElement("ul",{ref:u,className:"dropdown__menu"},t.map(((e,n)=>c.createElement(s.c,(0,a.c)({isDropdownItem:!0,onKeyDown:e=>{if(n===t.length-1&&"Tab"===e.key){e.preventDefault(),b(!1);const t=d.current.nextElementSibling;t&&t.focus()}},activeClassName:i},e,{key:n}))))))}function u(e){let{items:t,className:n,position:i,...d}=e;const u=(0,l.g5)(),h=m(t,u),{collapsed:b,toggleCollapsed:g,setCollapsed:v}=(0,l.au)({initialState:()=>!h});return(0,c.useEffect)((()=>{h&&v(!h)}),[u,h]),c.createElement("li",{className:(0,r.c)("menu__list-item",{"menu__list-item--collapsed":b})},c.createElement(o.A,(0,a.c)({role:"button",className:(0,r.c)("menu__link menu__link--sublist",n)},d,{onClick:e=>{e.preventDefault(),g()}}),d.children??d.label),c.createElement(l.Uv,{lazy:!0,as:"ul",className:"menu__list",collapsed:b},t.map(((e,t)=>c.createElement(s.c,(0,a.c)({mobile:!0,isDropdownItem:!0,onClick:d.onClick,activeClassName:"menu__link--active"},e,{key:t}))))))}const h=function(e){let{mobile:t=!1,...n}=e;const a=t?u:d;return c.createElement(a,n)}},5384:(e,t,n)=>{n.d(t,{c:()=>f,e:()=>v});var a=n(1504),c=n(9428),r=n(40),l=n(5072);const o=e=>{let{width:t=20,height:n=20,...c}=e;return a.createElement("svg",(0,l.c)({viewBox:"0 0 20 20",width:t,height:n,"aria-hidden":"true"},c),a.createElement("path",{fill:"currentColor",d:"M19.753 10.909c-.624-1.707-2.366-2.726-4.661-2.726-.09 0-.176.002-.262.006l-.016-2.063 3.525-.607c.115-.019.133-.119.109-.231-.023-.111-.167-.883-.188-.976-.027-.131-.102-.127-.207-.109-.104.018-3.25.461-3.25.461l-.013-2.078c-.001-.125-.069-.158-.194-.156l-1.025.016c-.105.002-.164.049-.162.148l.033 2.307s-3.061.527-3.144.543c-.084.014-.17.053-.151.143.019.09.19 1.094.208 1.172.018.08.072.129.188.107l2.924-.504.035 2.018c-1.077.281-1.801.824-2.256 1.303-.768.807-1.207 1.887-1.207 2.963 0 1.586.971 2.529 2.328 2.695 3.162.387 5.119-3.06 5.769-4.715 1.097 1.506.256 4.354-2.094 5.98-.043.029-.098.129-.033.207l.619.756c.08.096.206.059.256.023 2.51-1.73 3.661-4.515 2.869-6.683zm-7.386 3.188c-.966-.121-.944-.914-.944-1.453 0-.773.327-1.58.876-2.156a3.21 3.21 0 011.229-.799l.082 4.277a2.773 2.773 0 01-1.243.131zm2.427-.553l.046-4.109c.084-.004.166-.01.252-.01.773 0 1.494.145 1.885.361.391.217-1.023 2.713-2.183 3.758zm-8.95-7.668a.196.196 0 00-.196-.145h-1.95a.194.194 0 00-.194.144L.008 16.916c-.017.051-.011.076.062.076h1.733c.075 0 .099-.023.114-.072l1.008-3.318h3.496l1.008 3.318c.016.049.039.072.113.072h1.734c.072 0 .078-.025.062-.076-.014-.05-3.083-9.741-3.494-11.04zm-2.618 6.318l1.447-5.25 1.447 5.25H3.226z"}))};var s=n(8264),i=n(3100);const m="iconLanguage_EbrZ";function d(e){let{mobile:t,dropdownItemsBefore:n,dropdownItemsAfter:c,...d}=e;const{i18n:{currentLocale:u,locales:h,localeConfigs:b}}=(0,s.c)(),g=(0,i.DP)();function v(e){return b[e].label}const f=[...n,...h.map((e=>{const t=`pathname://${g.createUrl({locale:e,fullyQualified:!1})}`;return{isNavLink:!0,label:v(e),to:t,target:"_self",autoAddBaseUrl:!1,className:e===u?"dropdown__link--active":"",style:{textTransform:"capitalize"}}})),...c],E=t?"Languages":v(u);return a.createElement(r.c,(0,l.c)({},d,{href:"#",mobile:t,label:a.createElement("span",null,a.createElement(o,{className:m}),a.createElement("span",null,E)),items:f}))}var u=n(2944);function h(e){let{mobile:t}=e;return t?null:a.createElement(u.c,null)}const b={default:()=>c.c,localeDropdown:()=>d,search:()=>h,dropdown:()=>r.c,docsVersion:()=>n(4168).c,docsVersionDropdown:()=>n(5692).c,doc:()=>n(3040).c},g=e=>{const t=b[e];if(!t)throw new Error(`No NavbarItem component found for type "${e}".`);return t()};const v=e=>e?"menu__link--active":"navbar__link--active";function f(e){let{type:t,...n}=e;const c=function(e,t){return e&&"default"!==e?e:t?"dropdown":"default"}(t,void 0!==n.items),r=g(c);return a.createElement(r,n)}},520:(e,t,n)=>{n.d(t,{c:()=>a});const a=n(1504).createContext(void 0)},280:(e,t,n)=>{n.d(t,{c:()=>i});var a=n(5072),c=n(1504),r=n(4971),l=n(3664),o=n(9464);const s={themedImage:"themedImage_TMUO","themedImage--light":"themedImage--light_4Vu1","themedImage--dark":"themedImage--dark_uzRr"},i=e=>{const t=(0,l.c)(),{isDarkTheme:n}=(0,o.c)(),{sources:i,className:m,alt:d="",...u}=e,h=t?n?["dark"]:["light"]:["light","dark"];return c.createElement(c.Fragment,null,h.map((e=>c.createElement("img",(0,a.c)({key:e,src:i[e],alt:d,className:(0,r.c)(s.themedImage,s[`themedImage--${e}`],m)},u)))))}},3336:(e,t,n)=>{n.d(t,{c:()=>l});var a=n(1504),c=n(8684);const r=()=>c.c.canUseDOM?{scrollX:window.pageXOffset,scrollY:window.pageYOffset}:null,l=function(e,t){void 0===t&&(t=[]);const n=(0,a.useRef)(r()),c=()=>{const t=r();e&&e(t,n.current),n.current=t};(0,a.useEffect)((()=>{const e={passive:!0};return c(),window.addEventListener("scroll",c,e),()=>window.removeEventListener("scroll",c,e)}),t)}},9464:(e,t,n)=>{n.d(t,{c:()=>r});var a=n(1504),c=n(520);const r=function(){const e=(0,a.useContext)(c.c);if(null==e)throw new Error('"useThemeContext" is used outside of "Layout" component. Please see https://docusaurus.io/docs/api/themes/configuration#usethemecontext.');return e}}}]);