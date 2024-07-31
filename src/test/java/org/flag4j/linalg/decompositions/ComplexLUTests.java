package org.flag4j.linalg.decompositions;

import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.linalg.decompositions.lu.ComplexLU;
import org.flag4j.linalg.decompositions.lu.LU;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class ComplexLUTests {

    static CNumber[][] aEntries;
    static CMatrix A;

    CMatrix L;
    CMatrix U;
    Matrix P;
    Matrix Q;

    CMatrix A_hat;

    static ComplexLU lu;

    static void setMatrices() {
        A = new CMatrix(aEntries);
    }


    @Test

    void testnoPivotTestCase() {
        lu = new ComplexLU(0);

        // --------------------- Sub-case 1 ---------------------
        aEntries = new CNumber[][]{
                {new CNumber("1.255+45.1i"), new CNumber("-99.24+0.024i"), new CNumber("9.5-8.6i")},
                {new CNumber("6.466"), new CNumber("8.4-2.45i"), new CNumber("0.0+34.5i")},
                {new CNumber("1.05-47.1i"), new CNumber("66.2+3.5i"), new CNumber("45.33-0.009i")}};
        setMatrices();

        lu.decompose(A);
        L = lu.getL();
        U = lu.getU();
        A_hat = L.mult(U);

        assertEquals(new CMatrix(A.shape), A.sub(A_hat).roundToZero());

        // --------------------- Sub-case 2 ---------------------
        aEntries = new CNumber[][]{
                {new CNumber("1.255+45.1i"), new CNumber("-99.24+0.024i")},
                {new CNumber("6.466"), new CNumber("8.4-2.45i")},
                {new CNumber("1.05-47.1i"), new CNumber("66.2+3.5i")}};
        setMatrices();

        lu.decompose(A);
        L = lu.getL();
        U = lu.getU();
        A_hat = L.mult(U);

        assertEquals(new CMatrix(A.shape), A.sub(A_hat).roundToZero());

        // --------------------- Sub-case 3 ---------------------
        aEntries = new CNumber[][]{
                {new CNumber("1.255+45.1i"), new CNumber("-99.24+0.024i"), new CNumber("9.5-8.6i")},
                {new CNumber("6.466"), new CNumber("8.4-2.45i"), new CNumber("0.0+34.5i")}};
        setMatrices();

        lu.decompose(A);
        L = lu.getL();
        U = lu.getU();
        A_hat = L.mult(U);

        assertEquals(new CMatrix(A.shape), A.sub(A_hat).roundToZero());
    }


    @Test

    void testpartialPivotTestCase() {
        lu = new ComplexLU();

        // --------------------- Sub-case 1 ---------------------
        aEntries = new CNumber[][]{
                {new CNumber("1.255+45.1i"), new CNumber("-99.24+0.024i"), new CNumber("9.5-8.6i")},
                {new CNumber("6.466"), new CNumber("8.4-2.45i"), new CNumber("0.0+34.5i")},
                {new CNumber("1.05-47.1i"), new CNumber("66.2+3.5i"), new CNumber("45.33-0.009i")}};
        setMatrices();

        lu.decompose(A);
        P = lu.getP().toDense();
        L = lu.getL();
        U = lu.getU();
        A_hat = P.T().mult(L).mult(U);

        assertEquals(new CMatrix(A.shape), A.sub(A_hat).roundToZero());

        // --------------------- Sub-case 2 ---------------------
        aEntries = new CNumber[][]{
                {new CNumber("1.255+45.1i"), new CNumber("-99.24+0.024i")},
                {new CNumber("6.466"), new CNumber("8.4-2.45i")},
                {new CNumber("1.05-47.1i"), new CNumber("66.2+3.5i")}};
        setMatrices();

        lu.decompose(A);
        P = lu.getP().toDense();
        L = lu.getL();
        U = lu.getU();
        A_hat = P.T().mult(L).mult(U);

        assertEquals(new CMatrix(A.shape), A.sub(A_hat).roundToZero());

        // --------------------- Sub-case 3 ---------------------
        aEntries = new CNumber[][]{
                {new CNumber("1.255+45.1i"), new CNumber("-99.24+0.024i"), new CNumber("9.5-8.6i")},
                {new CNumber("6.466"), new CNumber("8.4-2.45i"), new CNumber("0.0+34.5i")}};
        setMatrices();

        lu.decompose(A);
        P = lu.getP().toDense();
        L = lu.getL();
        U = lu.getU();
        A_hat = P.T().mult(L).mult(U);

        assertEquals(new CMatrix(A.shape), A.sub(A_hat).roundToZero());
    }


    @Test
    void partialFullTestCase() {
        lu = new ComplexLU(LU.Pivoting.FULL.ordinal());

        // --------------------- Sub-case 1 ---------------------
        aEntries = new CNumber[][]{
                {new CNumber("1.255+45.1i"), new CNumber("-99.24+0.024i"), new CNumber("9.5-8.6i")},
                {new CNumber("6.466"), new CNumber("8.4-2.45i"), new CNumber("0.0+34.5i")},
                {new CNumber("1.05-47.1i"), new CNumber("66.2+3.5i"), new CNumber("45.33-0.009i")}};
        setMatrices();

        lu.decompose(A);
        P = lu.getP().toDense();
        Q = lu.getQ().toDense();
        L = lu.getL();
        U = lu.getU();
        A_hat = P.T().mult(L).mult(U).mult(Q.T());

        assertEquals(new CMatrix(A.shape), A.sub(A_hat).roundToZero());

        // --------------------- Sub-case 2 ---------------------
        aEntries = new CNumber[][]{
                {new CNumber("1.255+45.1i"), new CNumber("-99.24+0.024i")},
                {new CNumber("6.466"), new CNumber("8.4-2.45i")},
                {new CNumber("1.05-47.1i"), new CNumber("66.2+3.5i")}};
        setMatrices();

        lu.decompose(A);
        P = lu.getP().toDense();
        Q = lu.getQ().toDense();
        L = lu.getL();
        U = lu.getU();
        A_hat = P.T().mult(L).mult(U).mult(Q.T());

        assertEquals(new CMatrix(A.shape), A.sub(A_hat).roundToZero());

        // --------------------- Sub-case 3 ---------------------
        aEntries = new CNumber[][]{
                {new CNumber("1.255+45.1i"), new CNumber("-99.24+0.024i"), new CNumber("9.5-8.6i")},
                {new CNumber("6.466"), new CNumber("8.4-2.45i"), new CNumber("0.0+34.5i")}};
        setMatrices();

        lu.decompose(A);
        P = lu.getP().toDense();
        Q = lu.getQ().toDense();
        L = lu.getL();
        U = lu.getU();
        A_hat = P.T().mult(L).mult(U).mult(Q.T());

        assertEquals(new CMatrix(A.shape), A.sub(A_hat).roundToZero());
    }
}
