package com.flag4j.linalg.decompositions;

import com.flag4j.Matrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class RealSVDTests {

    final SingularValueDecomposition<Matrix> svd = new RealSVD();

    double[][] aEntries, expSEntries, expUEntries, expVEntries;

    Matrix A, expS, expU, expV;

    private void printJava(String name, Matrix src) {
        System.out.println(name + " = new double[][]{");

        for(int i=0; i<src.numRows; i++) {
            System.out.print("\t\t{");
            for(int j=0; j<src.numCols; j++) {
                System.out.print(src.get(i, j));
                if(j<src.numCols-1) {
                    System.out.print(", ");
                }
            }
            System.out.print("}");
            if(i<src.numRows-1) {
                System.out.print(", ");
            }
            System.out.println();
        }

        System.out.println("};");
        System.out.println(name.replace("Entries", "") + " = new Matrix(" + name + ");");
    }


    @Test
    void svdTest() {
        // -------------------- Sub-case 1 --------------------
        aEntries = new double[][]{
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9}
        };
        A = new Matrix(aEntries);

        expUEntries = new double[][]{
                {-0.21483723836839624, 0.8872306883463722, 4.3486798128278985E-7},
                {-0.520587389464737, 0.2496439529882979, 1.3380553270239688E-7},
                {-0.8263375405610778, -0.3879427823697761, -6.690276635119844E-8}
        };
        expU = new Matrix(expUEntries);
        expSEntries = new double[][]{
                {16.848103352614206, 0.0, 0.0},
                {0.0, 1.0683695145547063, 0.0},
                {0.0, 0.0, 6.637830303142728E-9}
        };
        expS = new Matrix(expSEntries);
        expVEntries = new double[][]{
                {-0.4796711778777717, -0.7766909903215606, 0.4082482904638607},
                {-0.5723677939720621, -0.07568647010455593, -0.816496580927726},
                {-0.6650644100663529, 0.6253180501124413, 0.40824829046386474}
        };
        expV = new Matrix(expVEntries);

        svd.decompose(A);
        assertEquals(expU, svd.getU());
        assertEquals(expS, svd.getS());
        assertEquals(expV, svd.getV());

        // -------------------- Sub-case 2 --------------------
        aEntries = new double[][]{
                {3.45, -99.34, 14.5, 24.5},
                {-0.0024, 0, 25.1, 1.5},
                {100.4, 5.6, -4.1, -0.002}
        };
        A = new Matrix(aEntries);
        expUEntries = new double[][]{
                {-0.9281923021372457, 0.3700008807116149, 0.03947655666076548},
                {-0.04026943724457484, 0.005584143138528899, -0.9991732531295128},
                {0.36991542638441793, 0.9290146207773262, -0.009716568571038788}
        };
        expU = new Matrix(expUEntries);
        expSEntries = new double[][]{
                {103.99805783777116, 0.0, 0.0, 0.0},
                {0.0, 100.10197909864624, 0.0, 0.0},
                {0.0, 0.0, 24.807778140450125, 0.0}
        };
        expS = new Matrix(expSEntries);
        expVEntries = new double[][]{
                {0.32632669031388206, 0.9445323500485516, -0.03373745699863598, -0.015209039937366702},
                {0.9065375992802964, -0.31521260516182376, -0.16027287490914857, 0.23052808984279557},
                {-0.15371637544368538, 0.016944867955476367, -0.9862632805045107, -0.05804213037879141},
                {-0.21925270397502236, 0.09062298112968553, -0.02142734449489553, 0.9712139805412126}
        };
        expV = new Matrix(expVEntries);

        svd.decompose(A);
        assertEquals(expU, svd.getU());
        assertEquals(expS, svd.getS());
        assertEquals(expV, svd.getV());

        // -------------------- Sub-case 3 --------------------
        aEntries = new double[][]{
                {34.5, 100.34},
                {-9.245, 0.13},
                {0, 1153.4},
                {14.5, -195.342}
        };
        A = new Matrix(aEntries);

        svd.decompose(A);
        printJava("expUEntries", svd.getU());
        printJava("expSEntries", svd.getS());
        printJava("expVEntries", svd.getV());
        System.out.print(svd.getU().mult(svd.getS()).mult(svd.getV().T()));
    }
}
