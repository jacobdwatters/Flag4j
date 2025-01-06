/**
 * Contains the core functionality of Flag4j (fast linear algebra for Java).
 */
module flag4j {
    requires java.logging;

    // Abstract algebra stuff.
    exports org.flag4j.algebraic_structures;

    // nD array stuff.
    exports org.flag4j.arrays;
    exports org.flag4j.arrays.dense;
    exports org.flag4j.arrays.sparse;

    // Concurrency.
    exports org.flag4j.concurrency;

    // I/O stuff.
    exports org.flag4j.io;

    // ------------------------- Linear algebra stuff -------------------------
    exports org.flag4j.linalg;

    // Decompositions.
    exports org.flag4j.linalg.decompositions;
    exports org.flag4j.linalg.decompositions.chol;
    exports org.flag4j.linalg.decompositions.hess;
    exports org.flag4j.linalg.decompositions.lu;
    exports org.flag4j.linalg.decompositions.qr;
    exports org.flag4j.linalg.decompositions.schur;
    exports org.flag4j.linalg.decompositions.svd;

    // Solvers.
    exports org.flag4j.linalg.solvers;
    exports org.flag4j.linalg.solvers.exact;
    exports org.flag4j.linalg.solvers.exact.triangular;
    exports org.flag4j.linalg.solvers.lstsq;

    // Transformations.
    exports org.flag4j.linalg.transformations;

    // Tensor Operations.
    exports org.flag4j.linalg.ops;
    exports org.flag4j.linalg.ops.common.complex;
    exports org.flag4j.linalg.ops.common.field_ops;
    exports org.flag4j.linalg.ops.common.real;
    exports org.flag4j.linalg.ops.common.ring_ops;
    exports org.flag4j.linalg.ops.common.semiring_ops;

    exports org.flag4j.linalg.ops.dense;
    exports org.flag4j.linalg.ops.dense.complex;
    exports org.flag4j.linalg.ops.dense.field_ops;
    exports org.flag4j.linalg.ops.dense.real;
    exports org.flag4j.linalg.ops.dense.real_field_ops;
    exports org.flag4j.linalg.ops.dense.semiring_ops;

    exports org.flag4j.linalg.ops.dense_sparse.coo.field_ops;
    exports org.flag4j.linalg.ops.dense_sparse.coo.real;
    exports org.flag4j.linalg.ops.dense_sparse.coo.real_complex;
    exports org.flag4j.linalg.ops.dense_sparse.coo.real_field_ops;

    exports org.flag4j.linalg.ops.dense_sparse.csr.field_ops;
    exports org.flag4j.linalg.ops.dense_sparse.csr.real;
    exports org.flag4j.linalg.ops.dense_sparse.csr.real_complex;
    exports org.flag4j.linalg.ops.dense_sparse.csr.real_field_ops;

    exports org.flag4j.linalg.ops.sparse;

    exports org.flag4j.linalg.ops.sparse.coo;
    exports org.flag4j.linalg.ops.sparse.coo.real;
    exports org.flag4j.linalg.ops.sparse.coo.real_complex;
    exports org.flag4j.linalg.ops.sparse.coo.ring_ops;
    exports org.flag4j.linalg.ops.sparse.coo.semiring_ops;

    exports org.flag4j.linalg.ops.sparse.csr;
    exports org.flag4j.linalg.ops.sparse.csr.real;
    exports org.flag4j.linalg.ops.sparse.csr.real_complex;
    // ------------------------------------------------------------------------

    // Random generation stuff.
    exports org.flag4j.rng;

    // Utilities
    exports org.flag4j.util;
    exports org.flag4j.util.exceptions;
    exports org.flag4j.linalg.ops.sparse.csr.ring_ops;
    exports org.flag4j.linalg.ops.dense_sparse.csr.semiring_ops;
}