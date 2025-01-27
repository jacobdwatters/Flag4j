package org.flag4j.arrays.sparse.sparse_complex_vector;

import org.flag4j.algebraic_structures.Complex128;
import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.arrays.sparse.CooVector;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CooCVectorUnaryOpTests {

    static int[] aIndices, expIndices;
    static Complex128[] aEntries, expEntries;
    double[] expEntriesRe;
    static int sparseSize;
    static CooCVector a, exp;
    CooVector expRe;


    @BeforeAll
    static void setup() {
        aEntries = new Complex128[]{
                new Complex128(2.455, -83.6), new Complex128(0, 24.56),
                new Complex128(24.56), new Complex128(-9356.1, 35)
        };
        aIndices = new int[]{4, 56, 9903, 14643};
        sparseSize = 24_023;
        a = new CooCVector(sparseSize, aEntries, aIndices);
    }


    @Test
    void hermTConjTestCase() {
        // ----------------------- sub-case 1 -----------------------
        expEntries = new Complex128[]{
                new Complex128(2.455, 83.6), new Complex128(0, -24.56),
                new Complex128(24.56), new Complex128(-9356.1, -35)
        };
        expIndices = new int[]{4, 56, 9903, 14643};
        exp = new CooCVector(sparseSize, expEntries, expIndices);

        assertEquals(exp, a.H());
        assertEquals(exp, a.conj());
    }


    @Test
    void transposeCopyTestCase() {
        // ----------------------- sub-case 1 -----------------------
        expEntries = new Complex128[]{
                new Complex128(2.455, -83.6), new Complex128(0, 24.56),
                new Complex128(24.56), new Complex128(-9356.1, 35)
        };
        expIndices = new int[]{4, 56, 9903, 14643};
        exp = new CooCVector(sparseSize, expEntries, expIndices);

        assertEquals(exp, a.T());
        assertEquals(exp, a.copy());
    }


    @Test
    void recipTestCase() {
        // ----------------------- sub-case 1 -----------------------
        expEntries = new Complex128[]{
                new Complex128(2.455, -83.6).multInv(), new Complex128(0, 24.56).multInv(),
                new Complex128(24.56).multInv(), new Complex128(-9356.1, 35).multInv()
        };
        expIndices = new int[]{4, 56, 9903, 14643};
        exp = new CooCVector(sparseSize, expEntries, expIndices);

        assertEquals(exp, a.recip());
    }


    @Test
    void absTestCase() {
        // ----------------------- sub-case 1 -----------------------
        expEntriesRe = new double[]{
                new Complex128(2.455, -83.6).mag(), new Complex128(0, 24.56).mag(),
                new Complex128(24.56).mag(), new Complex128(-9356.1, 35).mag()
        };
        expIndices = new int[]{4, 56, 9903, 14643};
        expRe = new CooVector(sparseSize, expEntriesRe, expIndices);

        assertEquals(expRe, a.abs());
    }


    @Test
    void sqrtTestCase() {
        // ----------------------- sub-case 1 -----------------------
        expEntries = new Complex128[]{
                new Complex128(2.455, -83.6).sqrt(), new Complex128(0, 24.56).sqrt(),
                new Complex128(24.56).sqrt(), new Complex128(-9356.1, 35).sqrt()
        };
        expIndices = new int[]{4, 56, 9903, 14643};
        exp = new CooCVector(sparseSize, expEntries, expIndices);

        assertEquals(exp, a.sqrt());
    }
}
