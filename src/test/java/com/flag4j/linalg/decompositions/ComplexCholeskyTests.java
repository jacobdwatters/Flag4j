package com.flag4j.linalg.decompositions;

import com.flag4j.dense.CMatrix;
import com.flag4j.linalg.decompositions.cholesky.ComplexCholeskyDecomposition;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ComplexCholeskyTests {

    static String[][] aEntries;
    static CMatrix A, L, A_hat;

    static void setMatrices() {
        A = new CMatrix(aEntries);
    }


    @Test
    void testcholeskyTestCase() {
        ComplexCholeskyDecomposition cholesky = new ComplexCholeskyDecomposition();

        // --------------------- Sub-case 1 ---------------------
        aEntries = new String[][]{
                {"1", "-2i"},
                {"2i", "5"}};
        setMatrices();

        L = cholesky.decompose(A).getL();
        A_hat = L.mult(L.H());

        assertEquals(new CMatrix(A.shape.copy()), A.sub(A_hat).roundToZero());

        // --------------------- Sub-case 2 ---------------------
        aEntries = new String[][]{
                {"2", "i"},
                {"-i", "2"}};
        setMatrices();

        L = cholesky.decompose(A).getL();
        A_hat = L.mult(L.H());

        assertEquals(new CMatrix(A.shape.copy()), A.sub(A_hat).roundToZero());
    }
}
