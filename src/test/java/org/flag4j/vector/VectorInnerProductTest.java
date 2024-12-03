package org.flag4j.vector;

import org.flag4j.algebraic_structures.fields.Complex128;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.linalg.ops.dense.real_field_ops.RealFieldDenseVectorOperations;
import org.flag4j.linalg.ops.dense_sparse.coo.real.RealDenseSparseVectorOperations;
import org.flag4j.linalg.ops.dense_sparse.coo.real_field_ops.RealFieldDenseCooVectorOps;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class VectorInnerProductTest {

    int[] indices;
    int sparseSize;
    double[] aEntries = {1.0, 5.6, -9.355, 215.0};
    Vector a = new Vector(aEntries);

    @Test
    void realDenseTestCase() {
        double[] bEntries;
        Vector b;
        Double exp;

        // ----------------------- Sub-case 1 -----------------------
        bEntries = new double[]{-0.2344, 0.0, 1.4667, 712.45};
        b = new Vector(bEntries);
        exp = 153162.7946215;

        assertEquals(exp, a.inner(b));
    }


    @Test
    void realSparseTestCase() {
        double[] bEntries;
        CooVector b;
        Double exp;

        // ----------------------- Sub-case 1 -----------------------
        bEntries = new double[]{1.4667};
        indices = new int[]{2};
        sparseSize = 4;
        b = new CooVector(sparseSize, bEntries, indices);
        exp = -13.7209785;

        assertEquals(exp, RealDenseSparseVectorOperations.inner(a.data, b.data, b.indices, b.size));
    }


    @Test
    void complexDenseTestCase() {
        Complex128[] bEntries;
        CVector b;
        Complex128 exp;

        // ----------------------- Sub-case 1 -----------------------
        bEntries = new Complex128[]{new Complex128("1.45-9.8i"), new Complex128("0.0+98.234i"),
                new Complex128("0.134"), new Complex128("-99.24+45.008i")};
        b = new CVector(bEntries);
        exp = new Complex128("-21336.40357-10217.030400000001j");

        assertEquals(exp, RealFieldDenseVectorOperations.inner(a.data, b.data));
    }


    @Test
    void complexSparseTestCase() {
        Complex128[] bEntries;
        CooCVector b;
        Complex128 exp;

        // ----------------------- Sub-case 1 -----------------------
        bEntries = new Complex128[]{new Complex128("1.45-9.8i"), new Complex128("-99.24+45.008i")};
        indices = new int[]{0, 3};
        sparseSize = 4;
        b = new CooCVector(sparseSize, bEntries, indices);
        exp = new Complex128("-21335.149999999998-9666.920000000002j");

        assertEquals(exp, RealFieldDenseCooVectorOps.inner(a.data, b.data, b.indices, b.size));
    }


    @Test
    void normalizeTestCase() {
        double[] expEntries;
        Vector exp;

        // ----------------------- Sub-case 1 -----------------------
        expEntries = new double[]{0.0046451435284722955, 0.026012803759444855, -0.043455317708858326, 0.9987058586215436};
        exp = new Vector(expEntries);

        assertEquals(exp, a.normalize());
    }
}
