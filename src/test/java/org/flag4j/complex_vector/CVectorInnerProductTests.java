package org.flag4j.complex_vector;

import org.flag4j.algebraic_structures.fields.Complex128;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.operations.dense.real_complex.RealComplexDenseVectorOperations;
import org.flag4j.operations.dense_sparse.coo.complex.ComplexDenseSparseVectorOperations;
import org.flag4j.operations.dense_sparse.coo.real_complex.RealComplexDenseSparseVectorOperations;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CVectorInnerProductTests {

    static Complex128[] aEntries;
    static CVector a;

    Complex128 exp;

    int[] sparseIndices;
    int sparseSize;

    @BeforeAll
    static void setup() {
        aEntries = new Complex128[]{new Complex128(1.455, 6126.347), new Complex128(-9.234, 5.0),
                new Complex128(9.245, -56.2345), new Complex128(0, 14.5), new Complex128(-0.009257)};
        a = new CVector(aEntries);
    }


    @Test()
    void realDenseInnerProdTestCase() {
        double[] bEntries;
        Vector b;

        // -------------------- Sub-case 1 --------------------
        bEntries = new double[]{1.455, 6.345, -0.00035, 1.56, -8815.56};
        b = new Vector(bEntries);
        exp = new Complex128("25.12969816999999 + 8968.199567075002i");

        assertEquals(exp, RealComplexDenseVectorOperations.inner(a.entries, b.entries));

        // -------------------- Sub-case 2 --------------------
        bEntries = new double[]{1.455, 1.56, -8815.56};
        b = new Vector(bEntries);

        Vector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->RealComplexDenseVectorOperations.inner(a.entries, finalB.entries));
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
        exp = new Complex128("-13.43870575+7.294682075000001j");

        assertEquals(exp, RealComplexDenseSparseVectorOperations.inner(a.entries, b.entries, b.indices, b.size));

        // -------------------- Sub-case 2 --------------------
        bEntries = new double[]{1.455, -0.00035};
        sparseIndices = new int[]{1, 2};
        sparseSize = 2;
        b = new CooVector(sparseSize, bEntries, sparseIndices);

        CooVector finalB = b;
        assertThrows(IllegalArgumentException.class,
                ()->RealComplexDenseSparseVectorOperations.inner(a.entries, finalB.entries, finalB.indices, finalB.size));
    }


    @Test()
    void complexDenseInnerProdTestCase() {
        Complex128[] bEntries;
        CVector b;

        // -------------------- Sub-case 1 --------------------
        bEntries = new Complex128[]{new Complex128(24.5, -6.01), new Complex128(3), new Complex128(0, 824),
            new Complex128(-9, 4.5), new Complex128(-0.00024, -5615.789)};
        b = new CVector(bEntries);
        exp = new Complex128("-83083.37796777832 + 142318.88069122698j");

        assertEquals(exp, a.inner(b));

        // -------------------- Sub-case 2 --------------------
        bEntries = new Complex128[]{new Complex128(24.5, -6.01), new Complex128(3), new Complex128(0, 824),
                new Complex128(-9, 4.5)};
        b = new CVector(bEntries);

        CVector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.inner(finalB));
    }


    @Test()
    void complexSparseInnerProdTestCase() {
        Complex128[] bEntries;
        CooCVector b;

        // -------------------- Sub-case 1 --------------------
        bEntries = new Complex128[]{new Complex128(24.5, -6.01), new Complex128(3)};
        sparseSize = 5;
        sparseIndices = new int[]{0, 4};
        b = new CooCVector(sparseSize, bEntries, sparseIndices);
        exp = new Complex128("-36783.725741 + 150104.24605j");

        assertEquals(exp, ComplexDenseSparseVectorOperations.innerProduct(a.entries, b.entries, b.indices, b.size));

        // -------------------- Sub-case 2 --------------------
        bEntries = new Complex128[]{new Complex128(24.5, -6.01), new Complex128(3)};
        sparseSize = 1145;
        sparseIndices = new int[]{0, 4};
        b = new CooCVector(sparseSize, bEntries, sparseIndices);

        CooCVector finalB = b;
        assertThrows(IllegalArgumentException.class,
                ()->ComplexDenseSparseVectorOperations.innerProduct(a.entries, finalB.entries, finalB.indices, finalB.size));
    }
}
