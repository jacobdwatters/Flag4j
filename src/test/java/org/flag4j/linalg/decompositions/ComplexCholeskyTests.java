package org.flag4j.linalg.decompositions;

import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.linalg.decompositions.chol.ComplexCholeskyOld;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ComplexCholeskyTests {

    static String[][] aEntries;
    static CMatrixOld A, L, A_hat;

    static void setMatrices() {
        A = new CMatrixOld(aEntries);
    }


    @Test
    void testcholeskyTestCase() {
        ComplexCholeskyOld cholesky = new ComplexCholeskyOld();

        // --------------------- Sub-case 1 ---------------------
        aEntries = new String[][]{
                {"1", "-2i"},
                {"2i", "5"}};
        setMatrices();

        L = cholesky.decompose(A).getL();
        A_hat = L.mult(L.H());

        assertEquals(new CMatrixOld(A.shape), A.sub(A_hat).roundToZero());

        // --------------------- Sub-case 2 ---------------------
        aEntries = new String[][]{
                {"2", "i"},
                {"-i", "2"}};
        setMatrices();

        L = cholesky.decompose(A).getL();
        A_hat = L.mult(L.H());

        assertEquals(new CMatrixOld(A.shape), A.sub(A_hat).roundToZero());
    }
}
