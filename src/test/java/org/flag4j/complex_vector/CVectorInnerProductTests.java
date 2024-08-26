package org.flag4j.complex_vector;

import org.flag4j.arrays_old.dense.CVectorOld;
import org.flag4j.arrays_old.dense.VectorOld;
import org.flag4j.arrays_old.sparse.CooCVectorOld;
import org.flag4j.arrays_old.sparse.CooVectorOld;
import org.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CVectorInnerProductTests {

    static CNumber[] aEntries;
    static CVectorOld a;

    CNumber exp;

    int[] sparseIndices;
    int sparseSize;

    @BeforeAll
    static void setup() {
        aEntries = new CNumber[]{new CNumber(1.455, 6126.347), new CNumber(-9.234, 5.0),
                new CNumber(9.245, -56.2345), new CNumber(0, 14.5), new CNumber(-0.009257)};
        a = new CVectorOld(aEntries);
    }


    @Test()
    void realDenseInnerProdTestCase() {
        double[] bEntries;
        VectorOld b;

        // -------------------- Sub-case 1 --------------------
        bEntries = new double[]{1.455, 6.345, -0.00035, 1.56, -8815.56};
        b = new VectorOld(bEntries);
        exp = new CNumber("25.12969816999999 + 8968.199567075002i");

        assertEquals(exp, a.inner(b));

        // -------------------- Sub-case 2 --------------------
        bEntries = new double[]{1.455, 1.56, -8815.56};
        b = new VectorOld(bEntries);

        VectorOld finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.inner(finalB));
    }


    @Test()
    void realSparseInnerProdTestCase() {
        double[] bEntries;
        CooVectorOld b;

        // -------------------- Sub-case 1 --------------------
        bEntries = new double[]{1.455, -0.00035};
        sparseIndices = new int[]{1, 2};
        sparseSize = 5;
        b = new CooVectorOld(sparseSize, bEntries, sparseIndices);
        exp = new CNumber("-13.43870575+7.294682075000001j");

        assertEquals(exp, a.inner(b));

        // -------------------- Sub-case 2 --------------------
        bEntries = new double[]{1.455, -0.00035};
        sparseIndices = new int[]{1, 2};
        sparseSize = 2;
        b = new CooVectorOld(sparseSize, bEntries, sparseIndices);

        CooVectorOld finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.inner(finalB));
    }


    @Test()
    void complexDenseInnerProdTestCase() {
        CNumber[] bEntries;
        CVectorOld b;

        // -------------------- Sub-case 1 --------------------
        bEntries = new CNumber[]{new CNumber(24.5, -6.01), new CNumber(3), new CNumber(0, 824),
            new CNumber(-9, 4.5), new CNumber(-0.00024, -5615.789)};
        b = new CVectorOld(bEntries);
        exp = new CNumber("-83083.37796777832 + 142318.88069122698j");

        assertEquals(exp, a.inner(b));

        // -------------------- Sub-case 2 --------------------
        bEntries = new CNumber[]{new CNumber(24.5, -6.01), new CNumber(3), new CNumber(0, 824),
                new CNumber(-9, 4.5)};
        b = new CVectorOld(bEntries);

        CVectorOld finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.inner(finalB));
    }


    @Test()
    void complexSparseInnerProdTestCase() {
        CNumber[] bEntries;
        CooCVectorOld b;

        // -------------------- Sub-case 1 --------------------
        bEntries = new CNumber[]{new CNumber(24.5, -6.01), new CNumber(3)};
        sparseSize = 5;
        sparseIndices = new int[]{0, 4};
        b = new CooCVectorOld(sparseSize, bEntries, sparseIndices);
        exp = new CNumber("-36783.725741 + 150104.24605j");

        assertEquals(exp, a.inner(b));

        // -------------------- Sub-case 2 --------------------
        bEntries = new CNumber[]{new CNumber(24.5, -6.01), new CNumber(3)};
        sparseSize = 1145;
        sparseIndices = new int[]{0, 4};
        b = new CooCVectorOld(sparseSize, bEntries, sparseIndices);

        CooCVectorOld finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.inner(finalB));
    }
}
