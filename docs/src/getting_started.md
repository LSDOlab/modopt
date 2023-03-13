# Getting started
This page provides instructions for installing your package 
and running a minimal example.

## Installation

### Installation instructions for users
For direct installation with all dependencies, run on the terminal or command line
```sh
pip install git+https://github.com/lsdolab/modopt.git
```
To update modopt to the latest version, run on the terminal or command line
```sh
pip uninstall modopt
pip install git+https://github.com/lsdolab/modopt.git
```

### Installation instructions for developers
To install `modopt`, first clone the repository and install using pip.
On the terminal or command line, run
```sh
git clone https://github.com/lsdolab/modopt.git
pip install -e ./modopt
```
To update modopt to the latest version, run on the terminal or command line
```none
cd /path/to/modopt
git pull
```

## Running a simple example
To test if your installation was successful, run 
`simple_example.py` in `modopt/modopt/examples`.
If everything works correctly, the following terminal output is
obtained.

```none
Setting objective name as "f".

----------------------------------------------------------------------------
Derivative type | Calc norm  | FD norm    | Abs error norm | Rel error norm 
----------------------------------------------------------------------------

Gradient        | 1.5274e-01 | 1.5274e-01 | 7.6367e-07     | 7.0710e-06    
----------------------------------------------------------------------------


         ===============================
         ModOpt final iteration summary:
         ===============================
         Problem       : x^4
         Solver        : steepest_descent
         itr           : 100
         obj           : 2.8304561425587446e-06
         opt           : 0.0002321105148388349
         time          : 0.8492448329925537
         =====================================


===============================
     modOpt summary table:     
===============================
 itr      obj      opt     time
   0 1.62E-02 1.53E-01 6.77E-05
  11 1.47E-04 4.49E-03 6.82E-02
  22 4.62E-05 1.88E-03 1.44E-01
  33 2.25E-05 1.10E-03 2.15E-01
  44 1.33E-05 7.40E-04 2.86E-01
  55 8.78E-06 5.43E-04 3.60E-01
  66 6.24E-06 4.20E-04 4.44E-01
  77 4.66E-06 3.37E-04 5.26E-01
  88 3.61E-06 2.79E-04 6.31E-01
 100 2.83E-06 2.32E-04 8.49E-01
===============================


100
[0.03449107 0.03449107]
0.8492448329925537
2.8304561425587446e-06
0.0002321105148388349
```