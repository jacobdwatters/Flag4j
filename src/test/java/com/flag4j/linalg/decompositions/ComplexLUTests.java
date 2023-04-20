package com.flag4j.linalg.decompositions;

import com.flag4j.CMatrix;
import com.flag4j.Matrix;
import com.flag4j.complex_numbers.CNumber;
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

    static ComplexLUDecomposition lu;

    static void setMatrices() {
        A = new CMatrix(aEntries);
    }


    @Test
    void noPivotTest() {
        lu = new ComplexLUDecomposition(0);

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

        assertEquals(new CMatrix(A.shape.copy()), A.sub(A_hat).roundToZero());

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

        assertEquals(new CMatrix(A.shape.copy()), A.sub(A_hat).roundToZero());

        // --------------------- Sub-case 3 ---------------------
        aEntries = new CNumber[][]{
                {new CNumber("1.255+45.1i"), new CNumber("-99.24+0.024i"), new CNumber("9.5-8.6i")},
                {new CNumber("6.466"), new CNumber("8.4-2.45i"), new CNumber("0.0+34.5i")}};
        setMatrices();

        lu.decompose(A);
        L = lu.getL();
        U = lu.getU();
        A_hat = L.mult(U);

        assertEquals(new CMatrix(A.shape.copy()), A.sub(A_hat).roundToZero());
    }


    @Test
    void partialPivotTest() {
        lu = new ComplexLUDecomposition();

        // --------------------- Sub-case 1 ---------------------
        aEntries = new CNumber[][]{
                {new CNumber("1.255+45.1i"), new CNumber("-99.24+0.024i"), new CNumber("9.5-8.6i")},
                {new CNumber("6.466"), new CNumber("8.4-2.45i"), new CNumber("0.0+34.5i")},
                {new CNumber("1.05-47.1i"), new CNumber("66.2+3.5i"), new CNumber("45.33-0.009i")}};
        setMatrices();

        lu.decompose(A);
        P = lu.getP();
        L = lu.getL();
        U = lu.getU();
        A_hat = P.T().mult(L).mult(U);

        assertEquals(new CMatrix(A.shape.copy()), A.sub(A_hat).roundToZero());

        // --------------------- Sub-case 2 ---------------------
        aEntries = new CNumber[][]{
                {new CNumber("1.255+45.1i"), new CNumber("-99.24+0.024i")},
                {new CNumber("6.466"), new CNumber("8.4-2.45i")},
                {new CNumber("1.05-47.1i"), new CNumber("66.2+3.5i")}};
        setMatrices();

        lu.decompose(A);
        P = lu.getP();
        L = lu.getL();
        U = lu.getU();
        A_hat = P.T().mult(L).mult(U);

        assertEquals(new CMatrix(A.shape.copy()), A.sub(A_hat).roundToZero());

        // --------------------- Sub-case 3 ---------------------
        aEntries = new CNumber[][]{
                {new CNumber("1.255+45.1i"), new CNumber("-99.24+0.024i"), new CNumber("9.5-8.6i")},
                {new CNumber("6.466"), new CNumber("8.4-2.45i"), new CNumber("0.0+34.5i")}};
        setMatrices();

        lu.decompose(A);
        P = lu.getP();
        L = lu.getL();
        U = lu.getU();
        A_hat = P.T().mult(L).mult(U);

        assertEquals(new CMatrix(A.shape.copy()), A.sub(A_hat).roundToZero());
    }


    @Test
    void partialFullTest() {
        lu = new ComplexLUDecomposition(LUDecomposition.Pivoting.FULL.ordinal());

        // --------------------- Sub-case 1 ---------------------
        aEntries = new CNumber[][]{
                {new CNumber("1.255+45.1i"), new CNumber("-99.24+0.024i"), new CNumber("9.5-8.6i")},
                {new CNumber("6.466"), new CNumber("8.4-2.45i"), new CNumber("0.0+34.5i")},
                {new CNumber("1.05-47.1i"), new CNumber("66.2+3.5i"), new CNumber("45.33-0.009i")}};
        setMatrices();

        lu.decompose(A);
        P = lu.getP();
        Q = lu.getQ();
        L = lu.getL();
        U = lu.getU();
        A_hat = P.T().mult(L).mult(U).mult(Q.T());

        assertEquals(new CMatrix(A.shape.copy()), A.sub(A_hat).roundToZero());

        // --------------------- Sub-case 2 ---------------------
        aEntries = new CNumber[][]{
                {new CNumber("1.255+45.1i"), new CNumber("-99.24+0.024i")},
                {new CNumber("6.466"), new CNumber("8.4-2.45i")},
                {new CNumber("1.05-47.1i"), new CNumber("66.2+3.5i")}};
        setMatrices();

        lu.decompose(A);
        P = lu.getP();
        Q = lu.getQ();
        L = lu.getL();
        U = lu.getU();
        A_hat = P.T().mult(L).mult(U).mult(Q.T());

        assertEquals(new CMatrix(A.shape.copy()), A.sub(A_hat).roundToZero());

        // --------------------- Sub-case 3 ---------------------
        aEntries = new CNumber[][]{
                {new CNumber("1.255+45.1i"), new CNumber("-99.24+0.024i"), new CNumber("9.5-8.6i")},
                {new CNumber("6.466"), new CNumber("8.4-2.45i"), new CNumber("0.0+34.5i")}};
        setMatrices();

        lu.decompose(A);
        P = lu.getP();
        Q = lu.getQ();
        L = lu.getL();
        U = lu.getU();
        A_hat = P.T().mult(L).mult(U).mult(Q.T());

        assertEquals(new CMatrix(A.shape.copy()), A.sub(A_hat).roundToZero());
    }
}
