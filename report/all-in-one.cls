\NeedsTeXFormat{LaTeX2e}

\RequirePackage{etoolbox}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\preto\@classoptionslist{english,}
\ProcessOptions\relax
\LoadClass{article}

\RequirePackage{ifthen}

\RequirePackage[utf8]{inputenc}
\RequirePackage{babel}
\RequirePackage[T1]{fontenc}

\RequirePackage{geometry}
\RequirePackage{calc}

\RequirePackage{amsmath}
\RequirePackage{mathtools}
\RequirePackage{amssymb}

\RequirePackage{amsthm}
\RequirePackage{thmtools}
\RequirePackage[svgnames]{xcolor}

\RequirePackage{graphicx}
\RequirePackage{setspace}
\RequirePackage{enumitem}
\RequirePackage{float}

\RequirePackage{titling}
\RequirePackage[explicit]{titlesec}
\RequirePackage{eso-pic, transparent}

\RequirePackage{extramarks}
\RequirePackage{fancyhdr}

\RequirePackage{xpatch}
\RequirePackage{rotating}
\RequirePackage{fbox}

\makeatletter
\newcommand{\doctitle}[1]{\def\@doctitle{#1}}
\newcommand{\docsubtitle}[1]{\def\@docsubtitle{#1}}
\newcommand{\footext}[1]{\def\@footext{#1}}
\makeatother

\numberwithin{equation}{section}
\mathtoolsset{showonlyrefs,showmanualtags}

\newcommand{\thrmname}{}
\newcommand{\corname}{}
\newcommand{\lemname}{}
\newcommand{\exname}{}
\newcommand{\defname}{}
\newcommand{\remname}{}
\newcommand{\exmpname}{}

\addto\captionsenglish{%
  \renewcommand{\thrmname}{Theorem}%
  \renewcommand{\corname}{Corollary}%
  \renewcommand{\lemname}{Lemma}%
  \renewcommand{\exname}{Exercise}%
  \renewcommand{\defname}{Definition}%
  \renewcommand{\remname}{Remark}%
  \renewcommand{\exmpname}{Example}%
  \renewcommand{\proofname}{Proof}%
}

\newtheorem{theorem}{\protect\thrmname}[section]
\newtheorem{corollary}[theorem]{\protect\corname}
\newtheorem{lemma}[theorem]{\protect\lemname}
\newtheorem{exercise}[theorem]{\protect\exname}

\theoremstyle{definition}
\newtheorem{definition}[theorem]{\protect\defname}

\theoremstyle{remark}
\newtheorem{remark}[theorem]{\protect\remname}
\newtheorem{example}[theorem]{\protect\exmpname}

\setlist{label=\textcolor{gray}{\textbullet},topsep=2pt,noitemsep}
\setenumerate{label=\textcolor{gray}{\textit{\arabic*}.},topsep=2pt,noitemsep}

\renewcommand{\headrulewidth}{1pt}
\xpretocmd\headrule{\color{lightgray}}{}{\PatchFailed}
\renewcommand{\qedsymbol}{\textcolor{gray}{$\blacksquare$}}

\geometry{a4paper, left=19.1mm, right=19.1mm, top=25.4mm, bottom=25.4mm}
\def\mylength{3pt}
\setlength{\headheight}{12pt+\mylength}
\addtolength{\headsep}{1em}
\addtolength{\footskip}{1em}
\addtolength{\textheight}{-2em-1ex-\mylength}

\AtBeginEnvironment{quote}{\itshape}

\newcommand\BackgroundPic{\put(0,0){\parbox[b][\paperheight]{\paperwidth}{\vfill\centering\includegraphics[width=\paperwidth,height=\paperheight,keepaspectratio]{img/fibonacci.pdf}\vfill}}}
\AtBeginEnvironment{titlepage}{\AddToShipoutPictureBG*{\transparent{0.2}\BackgroundPic}}

\newcommand{\pagenumber}[1]{{\setlength{\fboxsep}{5pt}\fbox[#1,lcolor=lightgray,rcolor=lightgray,tcolor=lightgray,bcolor=lightgray]{\fontsize{10pt}{10pt}\selectfont{\textcolor{gray!70!black}{\thepage}}}}}

\makeatletter

\fancypagestyle{singleside}{%
\fancyhead{}%
\lhead{\color{black}\nouppercase{\textsc{\rightmark}}}%
\rhead{\color{black}\nouppercase{\textsc{\lastrightxmark}}}%
\fancyfoot{}%
\rfoot{\pagenumber{rt}}%
\lfoot{\fontsize{10pt}{10pt}\selectfont{\textsc{\textcolor{black}{\@doctitle}\;\textcolor{lightgray}{|}\;\textcolor{gray!70!black}{\@docsubtitle}}}}%
\cfoot{\color{gray!70!black}\fontsize{10pt}{10pt}{\@footext}}%
\renewcommand{\headrulewidth}{1pt}}%

\fancypagestyle{twoside}{%
\fancyhead{}%
\fancyhead[LE]{\color{black}\nouppercase{\textsc{\rightmark}}}%
\fancyhead[RO]{\color{black}\nouppercase{\textsc{\rightmark}}}%
\fancyfoot{}%
\fancyfoot[RO]{\pagenumber{rt}}%
\fancyfoot[LE]{\pagenumber{lt}}%
\fancyfoot[RE]{\fontsize{10pt}{10pt}\selectfont{\textcolor{black}{\textsc{\@doctitle}}}}%
\fancyfoot[LO]{\fontsize{10pt}{10pt}\selectfont{\textcolor{gray!70!black}{\textsc{\@docsubtitle}}}}%
\fancyfoot[CO,CE]{\color{gray!70!black}\selectfont{\fontsize{10pt}{10pt}{\@footext}}}%
\renewcommand{\headrulewidth}{1pt}}%

\fancypagestyle{auxsingle}{%
\fancyhead{}%
\chead{\nouppercase{\textsc{\rightmark}}}%
\fancyfoot{}%
\lfoot{\fontsize{10pt}{10pt}\selectfont{\textsc{\textcolor{black}{\@doctitle}\;\textcolor{lightgray}{|}\;\textcolor{gray!70!black}{\@docsubtitle}}}}%
\rfoot{\pagenumber{rt}}%
\renewcommand{\headrulewidth}{1pt}}%

\fancypagestyle{auxtwoside}{%
\fancyhead{}%
\fancyhead[CE,CO]{\nouppercase{\textsc{\rightmark}}}%
\fancyfoot{}%
\fancyfoot[RE]{\fontsize{10pt}{10pt}\selectfont{\textcolor{black}{\textsc{\@doctitle}}}}%
\fancyfoot[LO]{\fontsize{10pt}{10pt}\selectfont{\textcolor{gray!70!black}{\textsc{\@docsubtitle}}}}%
\fancyfoot[RO]{\pagenumber{rt}}%
\fancyfoot[LE]{\pagenumber{lt}}%
\renewcommand{\headrulewidth}{1pt}}%

\makeatother

\makeatletter
\if@twoside
  \newcommand{\defaultsettings}{twoside}
  \newcommand{\auxsettings}{auxtwoside}
\else
  \newcommand{\defaultsettings}{singleside}
  \newcommand{\auxsettings}{auxsingle}
\fi
\makeatother

\pagestyle{\defaultsettings}

\renewcommand{\sectionmark}[1]{\markright{#1}}
\renewcommand{\subsectionmark}[1]{\extramarks{\thesubsection}{#1}}

\titleformat{\section}[frame]
{\color{gray}\large\normalfont\mdseries\scshape\centering}{\filcenter\small\quad{\color{black}\textit{\thesection}}\quad}
{1ex}{\color{gray!50!black}\Large\bfseries\filcenter#1}[]
\titleformat{\subsection}[hang]{\color{gray}\normalfont\mdseries\scshape\centering}{\textit{\thesubsection}}{2ex}{\color{gray!50!black}#1\hfill}
\titleformat{\subsubsection}[hang]
{\color{gray}\normalfont\mdseries\scshape\centering}
{\textit{\thesubsubsection}}
{2ex}{\color{gray!50!black}#1\hfill}