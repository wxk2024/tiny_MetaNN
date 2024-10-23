//
// Created by wxk on 2024/10/23.
//

#ifndef ALLOCATORS_H
#define ALLOCATORS_H
#include <MetaNN/data/facilities/tags.h>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <deque>

namespace MetaNN {
    template<typename TDevice>
    struct Allocator;

    // 针对 CPU 的特化，告诉我们应该怎样分配内存
    template<>
    struct Allocator<DeviceTags::CPU> {
    private:
        // 回收池
        struct AllocHelper {
            // 管理着大小和对应的内存
            std::unordered_map<size_t, std::deque<void *> > memBuffer;

            ~AllocHelper() {
                for (auto &p: memBuffer) {
                    auto &refVec = p.second;
                    for (auto &p1: refVec) {
                        char *buf = (char *) (p1);
                        delete []buf;
                    }
                    refVec.clear();
                }
            }
        };

        // 回收器
        struct DesImpl {
            DesImpl(std::deque<void *> &p_refPool): m_refPool(p_refPool) {
            }

            // 具体的回收操作
            void operator()(void *p_val) const {
                std::lock_guard<std::mutex> guard(GetMutex());
                m_refPool.push_back(p_val);
            }

        private:
            std::deque<void *> &m_refPool;
        };

    public:
        template<typename T>
        static std::shared_ptr<T> Allocate(size_t p_elemSize) {
            if (p_elemSize == 0) {
                return nullptr;
            }
            p_elemSize = (p_elemSize * sizeof(T) + 1023) & (size_t(-1) ^ 1023);
            std::lock_guard<std::mutex> guard(GetMutex());
            static AllocHelper allocateHelper;
            // 如果第一次没有，会自动创建一个池
            auto &slot = allocateHelper.memBuffer[p_elemSize];
            if (slot.empty()) {
                auto raw_buf = (T *) new char(p_elemSize);
                return std::shared_ptr<T>(raw_buf, DesImpl(slot));
            } else {
                void *mem = slot.back();
                slot.pop_back();
                return std::shared_ptr<T>((T *) mem, DesImpl(slot));
            }
        }

    private:
        static std::mutex &GetMutex() {
            static std::mutex m;
            return m;
        }
    };
}
#endif //ALLOCATORS_H
