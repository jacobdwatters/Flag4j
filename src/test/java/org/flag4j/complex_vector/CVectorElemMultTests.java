package org.flag4j.complex_vector;

import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CVectorElemMultTests {

    static CNumber[] aEntries;
    static CVector a;
    CNumber[] expEntries;
    CVector exp;

    int[] sparseIndices;
    int sparseSize;


    @BeforeAll
    static void setup() {
        aEntries = new CNumber[]{
                new CNumber(4.556, -85.2518), new CNumber(43.1, -99.34551),
                new CNumber(6915.66), new CNumber(0, 9.345)};
        a = new CVector(aEntries);
    }


    @Test
    void realDenseTestCase() {
        double[] bEntries;
        Vector b;

        // ------------------- Sub-case 1 -------------------
        bEntries = new double[]{2.455, -9.24, 0, 24.50001};
        b = new Vector(bEntries);
        expEntries = new CNumber[]{aEntries[0].mult(bEntries[0]), aEntries[1].mult(bEntries[1]),
                aEntries[2].mult(bEntries[2]), aEntries[3].mult(bEntries[3])};
        exp = new CVector(expEntries);

        assertEquals(exp, a.elemMult(b));

        // ------------------- Sub-case 2 -------------------
        bEntries = new double[]{2.455, -9.24};
        b = new Vector(bEntries);

        Vector finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.elemMult(finalB));
    }


    @Test
    void realSparseTestCase() {
        double[] bEntries;
        CooVector b;

        // ------------------- Sub-case 1 -------------------
        bEntries = new double[]{2.455, 24.50001};
        sparseIndices = new int[]{0, 1};
        sparseSize = 4;
        b = new CooVector(sparseSize, bEntries, sparseIndices);
        expEntries = new CNumber[]{aEntries[0].mult(bEntries[0]), aEntries[1].mult(bEntries[1]),
                CNumber.ZERO, CNumber.ZERO};
        exp = new CVector(expEntries);

        assertEquals(exp, a.elemMult(b).toDense());

        // ------------------- Sub-case 2 -------------------
        bEntries = new double[]{2.455, 24.50001};
        sparseIndices = new int[]{1, 3};
        sparseSize = 4;
        b = new CooVector(sparseSize, bEntries, sparseIndices);
        expEntries = new CNumber[]{CNumber.ZERO, aEntries[1].mult(bEntries[0]),
                CNumber.ZERO, aEntries[3].mult(bEntries[1])};
        exp = new CVector(expEntries);

        assertEquals(exp, a.elemMult(b).toDense());

        // ------------------- Sub-case 3 -------------------
        bEntries = new double[]{2.455, 24.50001};
        sparseIndices = new int[]{0, 1};
        sparseSize = 185234;
        b = new CooVector(sparseSize, bEntries, sparseIndices);

        CooVector finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.elemMult(finalB));
    }


    @Test
    void complexDenseTestCase() {
        CNumber[] bEntries;
        CVector b;

        // ------------------- Sub-case 1 -------------------
        bEntries = new CNumber[]{new CNumber(-0.00024), new CNumber(0, 85.234),
            new CNumber(0.00234, 15.6), new CNumber(-0.24, 662.115)};
        b = new CVector(bEntries);
        expEntries = new CNumber[]{aEntries[0].mult(bEntries[0]), aEntries[1].mult(bEntries[1]),
                aEntries[2].mult(bEntries[2]), aEntries[3].mult(bEntries[3])};
        exp = new CVector(expEntries);

        assertEquals(exp, a.elemMult(b));

        // ------------------- Sub-case 2 -------------------
        bEntries = new CNumber[]{new CNumber(0, 85.234),
                new CNumber(0.00234, 15.6), new CNumber(-0.24, 662.115)};
        b = new CVector(bEntries);

        CVector finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.elemMult(finalB));
    }

    @Test
    void complexSparseTestCase() {
        CNumber[] bEntries;
        CooCVector b;

        // ------------------- Sub-case 1 -------------------
        bEntries = new CNumber[]{new CNumber(234.566, -9.225), new CNumber(0.00024, 15.5)};
        sparseIndices = new int[]{0, 1};
        sparseSize = 4;
        b = new CooCVector(sparseSize, bEntries, sparseIndices);
        expEntries = new CNumber[]{aEntries[0].mult(bEntries[0]), aEntries[1].mult(bEntries[1]),
                CNumber.ZERO, CNumber.ZERO};
        exp = new CVector(expEntries);

        assertEquals(exp.toCoo(), a.elemMult(b));

        // ------------------- Sub-case 2 -------------------
        bEntries = new CNumber[]{new CNumber(-23.566, 0), new CNumber(0, 15.5)};
        sparseIndices = new int[]{1, 3};
        sparseSize = 4;
        b = new CooCVector(sparseSize, bEntries, sparseIndices);
        expEntries = new CNumber[]{CNumber.ZERO, aEntries[1].mult(bEntries[0]),
                CNumber.ZERO, aEntries[3].mult(bEntries[1])};
        exp = new CVector(expEntries);

        assertEquals(exp.toCoo(), a.elemMult(b));

        // ------------------- Sub-case 3 -------------------
        bEntries = new CNumber[]{new CNumber(-23.566, 0), new CNumber(0, 15.5)};
        sparseIndices = new int[]{0, 1};
        sparseSize = 185234;
        b = new CooCVector(sparseSize, bEntries, sparseIndices);

        CooCVector finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.elemMult(finalB));
    }
}
