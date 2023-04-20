package com.flag4j.linalg.decompositions;

import com.flag4j.Matrix;
import com.flag4j.linalg.decompositions.RealQRDecomposition;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class RealQRTests {

    double[][] aEntries, expQEntries, expREntries;
    Matrix A, expQ, expR;
    RealQRDecomposition QR;

    @Test
    void fullTest() {
        // Tests account for numerical loss of precision.
        QR = new RealQRDecomposition();

        // --------------------------- Sub-case 1 ---------------------------
        aEntries = new double[][]
                {{0, 1, 0, 0},
                {0, 0, 1, 0},
                {0, 0, 0, 1},
                {1, 0, 0, 0}};
        A = new Matrix(aEntries);
        expQEntries = new double[][]
                {{0.0, 1.0, 0.0, 0.0},
                {0.0, 0.0, 1.0, 0.0},
                {0.0, 0.0, 0.0, 1.0},
                {1.0, 0.0, 0.0, 0.0}};
        expQ = new Matrix(expQEntries);
        expREntries = new double[][]
                {{1.0, 0.0, 0.0, 0.0},
                {0.0, 1.0, 0.0, 0.0},
                {0.0, 0.0, 1.0, 0.0},
                {0.0, 0.0, 0.0, 1.0}};
        expR = new Matrix(expREntries);

        QR.decompose(A);

        assertEquals(expQ, QR.getQ());
        assertEquals(expR, QR.getR());

        // --------------------------- Sub-case 2 ---------------------------
        aEntries = new double[][]{{1.0, 5.6, -9.355, 215.0},
                {56.0, 1.0, 15.2, 14.0},
                {2.4, -0.00025, 1.0, 0.0},
                {1.0, 49.4, 106.2, -8.5}};
        A = new Matrix(aEntries);
        expQEntries = new double[][]{{0.017835090116782193, 0.11199486596156988, 0.9933393762756556, 0.02039468310780723},
                {0.9987650465398177, -0.01966789932383532, -0.014829165852483702, -0.043146858363307526},
                {0.042804216280277904, -0.0017104833343519145, -0.021083705815527223, 0.9988595249906157},
                {0.01783509011678246, 0.9935126561756523, -0.1123051945793043, -0.0014334747400640796}};
        expQ = new Matrix(expQEntries);
        expREntries = new double[][]{{56.06924290553602, 1.9796843019087813, 16.951272226045305, 17.66565676067297},
                {-4.449469684951587E-16, 49.68702899275902, 104.16266956172714, 15.358688013710783},
                {-1.1570602249277956E-15, 7.991680363279507E-15, -21.465988556154155, 214.3149517312553},
                {-7.765915587962469E-16, -1.7761944165101965E-16, -6.661338147750939E-16, 3.7929853863827914}};
        expR = new Matrix(expREntries);

        QR.decompose(A);

        assertEquals(expQ, QR.getQ());
        assertEquals(expR, QR.getR());
    }
}
