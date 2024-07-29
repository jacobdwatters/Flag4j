package org.flag4j.sparse_csr_complex_matrix;

import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.sparse.CsrCMatrix;
import org.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CsrCMatrixToDenseTests {
    static CsrCMatrix A;
    static CNumber[][] aEntries;

    static CMatrix exp;

    static void build() {
        A = new CMatrix(aEntries).toCsr();
        exp = new CMatrix(aEntries);
    }


    @Test
    void toDenseTests() {
        // --------------------- Sub-case 1 ---------------------
        aEntries = new CNumber[][]{
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.62854+0.22343i"), new CNumber("0.44136+0.15692i"), new CNumber("0.69915+0.87818i"), new CNumber("0.74834+0.32125i")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.27069+0.73057i"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.7623+0.12432i"), new CNumber("0.0")}};
        build();

        assertEquals(exp, A.toDense());

        // --------------------- Sub-case 2 ---------------------
        aEntries = new CNumber[][]{
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"),
                        new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.3778+0.2475i"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"),
                        new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.60458+0.65914i"), new CNumber("0.0"), new CNumber("0.08514+0.96172i"), new CNumber("0.0"),
                        new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.35664+0.90759i"), new CNumber("0.18187+0.30447i"), new CNumber("0.0"),
                        new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.67137+0.11747i")}};
        build();

        assertEquals(exp, A.toDense());
    }
}
