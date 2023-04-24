package com.flag4j.linalg.decompositions;

import com.flag4j.Matrix;
import com.flag4j.io.PrintOptions;
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
                {{0.0, -1.0, 0.0, 0.0},
                {0.0, 0.0, -1.0, 0.0},
                {0.0, 0.0, 0.0, -1.0},
                {-1.0, 0.0, 0.0, 0.0}};
        expQ = new Matrix(expQEntries);
        expREntries = new double[][]
                {{-1.0, 0.0, 0.0, 0.0},
                {0.0, -1.0, 0.0, 0.0},
                {0.0, 0.0, -1.0, 0.0},
                {0.0, 0.0, 0.0, -1.0}};
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
        expQEntries = new double[][]{
                {-0.017835090116782526, 0.11199486596156989, 0.9933393762756557, -0.02039468310780728},
                {-0.9987650465398175, -0.01966789932383531, -0.014829165852483533, 0.043146858363307505},
                {-0.04280421628027789, -0.0017104833343519062, -0.021083705815527258, -0.9988595249906157},
                {-0.017835090116782457, 0.9935126561756528, -0.11230519457930432, 0.0014334747400640679}};
        expQ = new Matrix(expQEntries);
        expREntries = new double[][]{
                {-56.06924290553601, -1.979684301908783, -16.951272226045297, -17.665656760673038},
                {9.445831541823943E-16, 49.687028992759025, 104.16266956172718, 15.35868801371079},
                {8.071648049173602E-15, 8.867225491518663E-15, -21.46598855615416, 214.31495173125532},
                {-3.5845519527462175E-16, -5.083331630562601E-16, -2.220446049250313E-16, -3.7929853863828047}};
        expR = new Matrix(expREntries);

        QR.decompose(A);

        assertEquals(expQ, QR.getQ());
        assertEquals(expR, QR.getR());
    }
}
