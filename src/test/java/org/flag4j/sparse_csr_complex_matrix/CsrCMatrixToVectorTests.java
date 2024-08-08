package org.flag4j.sparse_csr_complex_matrix;

import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.arrays.sparse.CsrCMatrix;
import org.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CsrCMatrixToVectorTests {
    CsrCMatrix A;
    CNumber[][] aEntries;

    CooCVector exp;
    CNumber[] expEntries;

    @Test
    void toVectorTests() {
        // ------------------------- Sub-case 1 -------------------------
        aEntries = new CNumber[][]{{new CNumber(1.23, -9.25), CNumber.ZERO, CNumber.ZERO,
                CNumber.ZERO, CNumber.ZERO, new CNumber(1.526, -3.1),
                CNumber.ZERO, CNumber.ZERO, new CNumber(0, 1.2)}};
        A = new CMatrix(aEntries).toCsr();
        expEntries = new CNumber[]{new CNumber(1.23, -9.25), CNumber.ZERO, CNumber.ZERO,
                CNumber.ZERO, CNumber.ZERO, new CNumber(1.526, -3.1),
                CNumber.ZERO, CNumber.ZERO, new CNumber(0, 1.2)};
        exp = new CVector(expEntries).toCoo();

        assertEquals(exp, A.toVector());

        // ------------------------- Sub-case 2 -------------------------
        aEntries = new CNumber[][]{{new CNumber(1.23, -9.25)}, {CNumber.ZERO}, {CNumber.ZERO},
                {CNumber.ZERO}, {CNumber.ZERO}, {new CNumber(1.526, -3.1)},
                {CNumber.ZERO}, {CNumber.ZERO}, {new CNumber(0, 1.2)}};
        A = new CMatrix(aEntries).toCsr();
        expEntries = new CNumber[]{new CNumber(1.23, -9.25), CNumber.ZERO, CNumber.ZERO,
                CNumber.ZERO, CNumber.ZERO, new CNumber(1.526, -3.1),
                CNumber.ZERO, CNumber.ZERO, new CNumber(0, 1.2)};
        exp = new CVector(expEntries).toCoo();

        assertEquals(exp, A.toVector());

        // ------------------------- Sub-case 3 -------------------------
        aEntries = new CNumber[][]{
                {new CNumber(1.23, -9.25), CNumber.ZERO},
                {CNumber.ZERO, CNumber.ZERO},
                {CNumber.ZERO, new CNumber(1.526, -3.1)},
                {CNumber.ZERO, CNumber.ZERO},
                {new CNumber(0, 1.2), CNumber.ZERO}};
        A = new CMatrix(aEntries).toCsr();
        expEntries = new CNumber[]{new CNumber(1.23, -9.25), CNumber.ZERO, CNumber.ZERO,
                CNumber.ZERO, CNumber.ZERO, new CNumber(1.526, -3.1),
                CNumber.ZERO, CNumber.ZERO, new CNumber(0, 1.2), CNumber.ZERO};
        exp = new CVector(expEntries).toCoo();

        assertEquals(exp, A.toVector());
    }
}
