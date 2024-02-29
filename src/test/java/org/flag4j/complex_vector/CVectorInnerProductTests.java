package org.flag4j.complex_vector;

import org.flag4j.complex_numbers.CNumber;
import org.flag4j.dense.CVector;
import org.flag4j.dense.Vector;
import org.flag4j.sparse.CooCVector;
import org.flag4j.sparse.CooVector;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CVectorInnerProductTests {

    static CNumber[] aEntries;
    static CVector a;

    CNumber exp;

    int[] sparseIndices;
    int sparseSize;

    @BeforeAll
    static void setup() {
        aEntries = new CNumber[]{new CNumber(1.455, 6126.347), new CNumber(-9.234, 5.0),
                new CNumber(9.245, -56.2345), new CNumber(0, 14.5), new CNumber(-0.009257)};
        a = new CVector(aEntries);
    }


    @Test()
    void realDenseInnerProdTestCase() {
        double[] bEntries;
        Vector b;

        // -------------------- Sub-case 1 --------------------
        bEntries = new double[]{1.455, 6.345, -0.00035, 1.56, -8815.56};
        b = new Vector(bEntries);
        exp = new CNumber("25.12969816999999 + 8968.199567075002i");

        assertEquals(exp, a.inner(b));

        // -------------------- Sub-case 2 --------------------
        bEntries = new double[]{1.455, 1.56, -8815.56};
        b = new Vector(bEntries);

        Vector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.inner(finalB));
    }


    @Test()
    void realSparseInnerProdTestCase() {
        double[] bEntries;
        CooVector b;

        // -------------------- Sub-case 1 --------------------
        bEntries = new double[]{1.455, -0.00035};
        sparseIndices = new int[]{1, 2};
        sparseSize = 5;
        b = new CooVector(sparseSize, bEntries, sparseIndices);
        exp = new CNumber("-13.43870575+7.294682075000001j");

        assertEquals(exp, a.inner(b));

        // -------------------- Sub-case 2 --------------------
        bEntries = new double[]{1.455, -0.00035};
        sparseIndices = new int[]{1, 2};
        sparseSize = 2;
        b = new CooVector(sparseSize, bEntries, sparseIndices);

        CooVector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.inner(finalB));
    }


    @Test()
    void complexDenseInnerProdTestCase() {
        CNumber[] bEntries;
        CVector b;

        // -------------------- Sub-case 1 --------------------
        bEntries = new CNumber[]{new CNumber(24.5, -6.01), new CNumber(3), new CNumber(0, 824),
            new CNumber(-9, 4.5), new CNumber(-0.00024, -5615.789)};
        b = new CVector(bEntries);
        exp = new CNumber("-83083.37796777832 + 142318.88069122698j");

        assertEquals(exp, a.inner(b));

        // -------------------- Sub-case 2 --------------------
        bEntries = new CNumber[]{new CNumber(24.5, -6.01), new CNumber(3), new CNumber(0, 824),
                new CNumber(-9, 4.5)};
        b = new CVector(bEntries);

        CVector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.inner(finalB));
    }


    @Test()
    void complexSparseInnerProdTestCase() {
        CNumber[] bEntries;
        CooCVector b;

        // -------------------- Sub-case 1 --------------------
        bEntries = new CNumber[]{new CNumber(24.5, -6.01), new CNumber(3)};
        sparseSize = 5;
        sparseIndices = new int[]{0, 4};
        b = new CooCVector(sparseSize, bEntries, sparseIndices);
        exp = new CNumber("-36783.725741 + 150104.24605j");

        assertEquals(exp, a.inner(b));

        // -------------------- Sub-case 2 --------------------
        bEntries = new CNumber[]{new CNumber(24.5, -6.01), new CNumber(3)};
        sparseSize = 1145;
        sparseIndices = new int[]{0, 4};
        b = new CooCVector(sparseSize, bEntries, sparseIndices);

        CooCVector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.inner(finalB));
    }
}
